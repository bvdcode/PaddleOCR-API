import io
import os
from typing import List, Dict, Any, Optional, Iterable
from PIL import Image
import pdf2image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import inspect
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PaddleOCR API",
              description="CPU-only OCR service", version="1.0.0")


def _filter_kwargs(callable_obj, desired: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
        accepted = {k: v for k, v in desired.items() if k in sig.parameters}
        dropped = set(desired) - set(accepted)
        if dropped:
            logger.info(
                f"Dropped unsupported args for {callable_obj.__name__}: {dropped}")
        return accepted
    except Exception:
        return desired


_OCR_CACHE: Dict[str, PaddleOCR] = {}


def create_ocr(lang: str) -> PaddleOCR:
    desired = dict(use_angle_cls=True, lang=lang,
                   show_log=False, use_gpu=False)
    filtered = _filter_kwargs(PaddleOCR, desired)
    try:
        inst = PaddleOCR(**filtered)
        logger.info(f"Initialized PaddleOCR lang={lang} args={filtered}")
        return inst
    except Exception as e:
        logger.warning(
            f"OCR init failed lang={lang} {filtered}: {e}; retry minimal")
        minimal = _filter_kwargs(PaddleOCR, dict(lang=lang))
        inst = PaddleOCR(**minimal)
        logger.info(
            f"Initialized PaddleOCR minimal lang={lang} args={minimal}")
        return inst


def get_ocr(lang: str) -> PaddleOCR:
    if lang not in _OCR_CACHE:
        _OCR_CACHE[lang] = create_ocr(lang)
    return _OCR_CACHE[lang]


_TABLE_ENGINE = None
_TABLE_ENGINE_ATTEMPTED = False  # sentinel to avoid repeated noisy logs


def get_table_engine():
    """Lazy init PPStructure once. No env flag required.

    Returns None if PPStructure not installed or init failed. Uses a sentinel
    to avoid re-attempt spam on every page.
    """
    global _TABLE_ENGINE, _TABLE_ENGINE_ATTEMPTED
    if _TABLE_ENGINE is not None or _TABLE_ENGINE_ATTEMPTED:
        return _TABLE_ENGINE
    try:
        from paddleocr import PPStructure
    except ImportError:
        logger.info("PP-Structure not installed; table extraction disabled")
        _TABLE_ENGINE_ATTEMPTED = True
        return None
    # Request table module; do not force ocr=False (can break pipeline). Use 'en' to avoid extra language-specific downloads.
    desired = dict(lang='en', show_log=False,
                   use_gpu=False, table=True, layout=True)
    filtered = _filter_kwargs(PPStructure, desired)
    try:
        engine = PPStructure(**filtered)
        logger.info(f"Initialized PPStructure for tables args={filtered}")
        _TABLE_ENGINE = engine
    except Exception as e:
        logger.warning(f"PPStructure init failed: {e}")
        _TABLE_ENGINE = None
    _TABLE_ENGINE_ATTEMPTED = True
    return _TABLE_ENGINE


def parse_pages(pages_str: str, total_pages: int) -> List[int]:
    if not pages_str:
        return list(range(1, total_pages + 1))
    pages: List[int] = []
    for part in pages_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return [p for p in pages if 1 <= p <= total_pages]


def pdf_to_images(pdf_bytes: bytes, dpi: int = 350) -> List[Image.Image]:
    try:
        # Performance note: High DPI (e.g., 350) increases accuracy but also
        # increases processing time significantly. For large PDFs consider
        # using 200-250 DPI or dynamically downscaling oversized pages after
        # rasterization to reduce end-to-end latency.
        return pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi, fmt='RGB')
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error processing PDF: {e}")


def _coerce_bbox(obj: Any) -> Optional[List[float]]:
    """Attempt to coerce various bbox / polygon representations into
    [x_min, y_min, x_max, y_max]. Accepts:
      - list/tuple of 4 numbers already in that form
      - list of points [[x,y], ...]
      - dict with keys x1,y1,x2,y2 or left,top,right,bottom
      - list of 8 numbers representing 4 (x,y) pairs
    Returns None if cannot parse.
    """
    try:
        # Direct 4 numeric values
        if isinstance(obj, (list, tuple)) and len(obj) == 4 and all(isinstance(v, (int, float)) for v in obj):
            return [float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])]
        # 8-number flattened polygon
        if isinstance(obj, (list, tuple)) and len(obj) == 8 and all(isinstance(v, (int, float)) for v in obj):
            xs = obj[0::2]
            ys = obj[1::2]
            return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        # List of points
        if isinstance(obj, (list, tuple)) and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in obj):
            xs = [p[0] for p in obj]
            ys = [p[1] for p in obj]
            if xs and ys:
                return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        # Dict formats
        if isinstance(obj, dict):
            if all(k in obj for k in ("x1", "y1", "x2", "y2")):
                return [float(obj['x1']), float(obj['y1']), float(obj['x2']), float(obj['y2'])]
            if all(k in obj for k in ("left", "top", "right", "bottom")):
                return [float(obj['left']), float(obj['top']), float(obj['right']), float(obj['bottom'])]
            if 'points' in obj:
                return _coerce_bbox(obj['points'])
            if 'poly' in obj:
                return _coerce_bbox(obj['poly'])
            if 'polygon' in obj:
                return _coerce_bbox(obj['polygon'])
            if 'bbox' in obj:
                return _coerce_bbox(obj['bbox'])
            if 'box' in obj:
                return _coerce_bbox(obj['box'])
        return None
    except Exception:
        return None


def extract_text_from_image(image: Image.Image, lang: str) -> Dict[str, Any]:
    """Run legacy .ocr() and extract text robustly.

    Adds:
      - Suppression of noisy stdout (unless OCR_DEBUG=1)
      - Multi-schema parsing (classic list, list-of-dict, deep recurse fallback)
      - Optional deep parse toggle (OCR_DEEP_PARSE=1)
      - Logging of counts only (no raw text) when OCR_LOG_COUNTS=1 (default)
    """
    try:
        ocr_engine = get_ocr(lang)
        arr = np.array(image.convert('RGB'))  # ensure RGB
        import contextlib
        import time
        import io as _io
        capture = _io.StringIO()
        debug = os.getenv('OCR_DEBUG', '0').lower() in {'1', 'true', 'yes'}
        t0 = time.time()
        if debug:
            try:
                result = ocr_engine.ocr(arr, cls=True)
            except TypeError:
                result = ocr_engine.ocr(arr)
        else:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                try:
                    result = ocr_engine.ocr(arr, cls=True)
                except TypeError:
                    result = ocr_engine.ocr(arr)
        elapsed = (time.time() - t0) * 1000

    # We'll collect structured line objects: {text, confidence, box, bbox}
        structured: List[Dict[str, Any]] = []
        lines_plain: List[str] = []

        # Pattern A: nested list (older format: [ [ [box,(text,conf)], ... ] ])
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, list) and all(isinstance(e, (list, tuple)) for e in first):
                for entry in first:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        box = entry[0]
                        txt_part = entry[1]
                        text_val: Optional[str] = None
                        conf_val: Optional[float] = None
                        if isinstance(txt_part, (list, tuple)) and txt_part:
                            text_val = str(txt_part[0])
                            if len(txt_part) > 1:
                                try:
                                    conf_val = float(txt_part[1])
                                except Exception:
                                    conf_val = None
                        elif isinstance(txt_part, str):
                            text_val = txt_part
                        if text_val:
                            lines_plain.append(text_val)
                            structured.append(
                                {"text": text_val, "confidence": conf_val, "box": box})
            # Pattern B: flat list directly of entries [ [box,(text,conf)], ... ]
            elif all(isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[0], (list, tuple)) for entry in result):
                for entry in result:
                    box = entry[0]
                    txt_part = entry[1]
                    text_val: Optional[str] = None
                    conf_val: Optional[float] = None
                    if isinstance(txt_part, (list, tuple)) and txt_part:
                        text_val = str(txt_part[0])
                        if len(txt_part) > 1:
                            try:
                                conf_val = float(txt_part[1])
                            except Exception:
                                conf_val = None
                    elif isinstance(txt_part, str):
                        text_val = txt_part
                    if text_val:
                        lines_plain.append(text_val)
                        structured.append(
                            {"text": text_val, "confidence": conf_val, "box": box})

        # Secondary: list of dicts with 'text'
        if not structured and isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and 'text' in item and isinstance(item['text'], str):
                    structured.append({
                        "text": item['text'],
                        "confidence": item.get('confidence') or item.get('score'),
                        "box": item.get('box') or item.get('bbox')
                    })
                    lines_plain.append(item['text'])

        # Tertiary: dict with 'data' or nested lists
        if not structured and isinstance(result, dict):
            data = result.get('data') or result.get('res')
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'text' in item and isinstance(item['text'], str):
                        structured.append({
                            "text": item['text'],
                            "confidence": item.get('confidence') or item.get('score'),
                            "box": item.get('box') or item.get('bbox')
                        })
                        lines_plain.append(item['text'])

        # Deep recursive fallback (guarded) if still empty
        deep_enabled = os.getenv('OCR_DEEP_PARSE', '1').lower() in {
            '1', 'true', 'yes'}
        if not structured and deep_enabled:
            collected: List[str] = []

            def visit(obj, depth=0):
                if len(collected) >= 500:
                    return
                if isinstance(obj, (list, tuple)):
                    for x in obj:
                        visit(x, depth+1)
                elif isinstance(obj, dict):
                    for k in ('text', 'transcription', 'value'):
                        v = obj.get(k)
                        if isinstance(v, str) and 1 <= len(v) <= 512:
                            collected.append(v)
                    for v in obj.values():
                        if isinstance(v, (list, tuple, dict)):
                            visit(v, depth+1)
                elif isinstance(obj, str):
                    if 1 <= len(obj) <= 64:
                        collected.append(obj)

            visit(result)
            seen = set()
            for t in collected:
                if t not in seen:
                    seen.add(t)
                    lines_plain.append(t)
                    structured.append(
                        {"text": t, "confidence": None, "box": None})

        # Derive normalized axis-aligned bbox for each line using universal coercion
        for line in structured:
            box_raw = line.get('box') or line.get('bbox')
            bbox = _coerce_bbox(box_raw)
            if bbox:
                line['bbox'] = bbox
        full_text = '\n'.join(lines_plain)
        # Confidence stats
        conf_values = [c['confidence'] for c in structured if isinstance(
            c.get('confidence'), (int, float))]
        stats = None
        if conf_values:
            stats = {
                'avg_confidence': sum(conf_values)/len(conf_values),
                'min_confidence': min(conf_values),
                'max_confidence': max(conf_values),
                'count_confidence': len(conf_values)
            }
        if os.getenv('OCR_LOG_COUNTS', '1').lower() in {'1', 'true', 'yes'}:
            logger.info(
                f"OCR page: lines={len(structured)} chars={len(full_text)} time_ms={elapsed:.1f}")
        # Always include truncated raw snippet for debugging even if not in full debug
        raw_str = None
        try:
            raw_str = str(result)
        except Exception:
            raw_str = None
        raw_truncated = raw_str[:1200] if raw_str else None
        if debug and not structured and raw_truncated:
            logger.debug(f"RAW OCR RESULT SAMPLE (trim): {raw_truncated}")
        return {"full_text": full_text, "lines": structured, "raw": result if debug else None, "raw_truncated": raw_truncated, "stats": stats}
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return {"full_text": "", "lines": []}


def extract_tables_from_image(image: Image.Image, debug: bool = False) -> Dict[str, Any]:
    """Extract tables using PP-Structure (if enabled) returning structured cell data only.

    Output per table:
      {
        'bbox': [...],
        'cells': [ {'text': str, 'bbox': [...]} ...],
        'cells_count': int
      }
    HTML intentionally removed. If PP-Structure models are not present they
    will be downloaded automatically on first use (if supported by install).
    """
    engine = get_table_engine()
    if not engine:
        return {"tables": [], "raw": None, "raw_truncated": None}
    try:
        arr = np.array(image.convert('RGB'))
        result = engine(arr)
        tables: List[Dict[str, Any]] = []
        total_cells = 0
        layout_blocks: List[Dict[str, Any]] = []
        if isinstance(result, list):
            # First pass: direct table items
            for item in result:
                if isinstance(item, dict) and item.get('type') == 'table':
                    res_block = item.get('res') or {}
                    raw_cells = res_block.get('cells') or []
                    if not raw_cells and 'cell' in res_block:
                        raw_cells = res_block['cell']
                    if not raw_cells and 'cells_list' in res_block:
                        raw_cells = res_block['cells_list']
                    cells_out: List[Dict[str, Any]] = []
                    for c in raw_cells:
                        if not isinstance(c, dict):
                            continue
                        txt = c.get('text') or c.get('cell_text') or ''
                        bbox = _coerce_bbox(c.get('bbox') or c.get(
                            'box') or c.get('poly') or c.get('polygon'))
                        cells_out.append({'text': txt, 'bbox': bbox})
                    total_cells += len(cells_out)
                    tables.append({'bbox': _coerce_bbox(item.get('bbox') or item.get('box') or item.get('poly') or item.get('polygon')) or [],
                                   'cells': cells_out, 'cells_count': len(cells_out)})
                elif isinstance(item, dict):
                    layout_blocks.append(item)
            # Fallback: inspect layout blocks for potential implicit tables (grid-like text)
            if not tables and layout_blocks:
                # Heuristic: group blocks that share similar y-ranges to form rows
                candidate_lines = []
                for b in layout_blocks:
                    if b.get('type') in {'text', 'figure', 'layout'}:
                        txt = b.get('res', {}).get('text') if isinstance(
                            b.get('res'), dict) else None
                        if isinstance(txt, str) and len(txt.strip()):
                            bbox = _coerce_bbox(b.get('bbox') or b.get(
                                'box') or b.get('poly') or b.get('polygon'))
                            if bbox:
                                candidate_lines.append(
                                    {'text': txt, 'bbox': bbox})
                # Simple row clustering by vertical overlap
                rows: List[List[Dict[str, Any]]] = []
                for line in sorted(candidate_lines, key=lambda x: x['bbox'][1]):
                    placed = False
                    for row in rows:
                        # if vertical intersection > 40% treat as same row
                        y_min = max(line['bbox'][1], min(
                            c['bbox'][1] for c in row))
                        y_max = min(line['bbox'][3], max(
                            c['bbox'][3] for c in row))
                        h = min(line['bbox'][3]-line['bbox'][1], max(c['bbox'][3]
                                for c in row)-min(c['bbox'][1] for c in row))
                        if h > 0 and (y_max - y_min) / h >= 0.4:
                            row.append(line)
                            placed = True
                            break
                    if not placed:
                        rows.append([line])
                # If we have multi-column style rows, treat as a single inferred table
                if rows and sum(len(r) for r in rows) >= 6 and any(len(r) >= 2 for r in rows):
                    # Flatten into cells
                    all_cells = []
                    x1 = min(min(c['bbox'][0] for c in r) for r in rows)
                    y1 = min(min(c['bbox'][1] for c in r) for r in rows)
                    x2 = max(max(c['bbox'][2] for c in r) for r in rows)
                    y2 = max(max(c['bbox'][3] for c in r) for r in rows)
                    for r in rows:
                        for c in r:
                            all_cells.append(
                                {'text': c['text'], 'bbox': c['bbox']})
                    tables.append({'bbox': [x1, y1, x2, y2], 'cells': all_cells, 'cells_count': len(
                        all_cells), 'inferred': True})
                    total_cells += len(all_cells)
        if (os.getenv('TABLE_DEBUG', '0').lower() in {'1', 'true', 'yes'}) or (os.getenv('OCR_DEBUG', '0').lower() in {'1', 'true', 'yes'}):
            logger.info(
                f"table detection: raw_items={len(result) if isinstance(result, list) else 'n/a'} tables={len(tables)} total_cells={total_cells}")
        raw_summary = None
        raw_str = None
        try:
            raw_str = str(result)
        except Exception:
            pass
        raw_truncated = raw_str[:1200] if raw_str else None
        if isinstance(result, list):
            raw_summary = [r.get('type', 'unknown') if isinstance(
                r, dict) else type(r).__name__ for r in result[:5]]
        return {"tables": tables, "raw": result if debug else None, "raw_summary": raw_summary, "raw_truncated": raw_truncated}
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        return {"tables": [], "raw": None, "raw_truncated": None, "error": str(e)}


def process_image(image: Image.Image, page_index: int, lang: str, do_tables: bool, debug: bool) -> Dict[str, Any]:
    """Run OCR + (optional) table detection for a single page.

    HTML output was removed per user request; only structured JSON is returned.
    """
    import time
    t0 = time.time()
    text_struct = extract_text_from_image(image, lang)
    ocr_ms = (time.time() - t0)*1000
    t1 = time.time()
    table_struct: Dict[str, Any] = {"tables": []}
    if do_tables:
        table_struct = extract_tables_from_image(image, debug)
    table_ms = (time.time() - t1)*1000 if do_tables else 0.0
    text_full = text_struct["full_text"]
    stats = text_struct.get('stats') or {}
    page: Dict[str, Any] = {
        "index": page_index,
        "text": {
            "full": text_full,
            "length": len(text_full),
            "lines": text_struct["lines"],
            "lines_count": len(text_struct["lines"])
        },
        "tables": table_struct.get("tables", []),
        "metrics": {
            "ocr_ms": round(ocr_ms, 2),
            "table_ms": round(table_ms, 2),
            "total_ms": round((time.time() - t0)*1000, 2),
            **({k: round(v, 6) for k, v in stats.items()} if stats else {})
        }
    }
    # Always include truncated raw snippets for transparency
    page["raw"] = {
        "ocr_truncated": text_struct.get("raw_truncated"),
        "tables_truncated": table_struct.get("raw_truncated")
    }
    if debug:
        page["debug"] = {
            "tables_raw_count": len(table_struct.get("raw", []) if isinstance(table_struct.get("raw"), list) else []),
        }
    return page


@app.get('/health')
async def health():
    return {"status": "healthy", "service": "PaddleOCR-API"}


@app.post('/analyze')
async def analyze(
    file: UploadFile = File(...),
    dpi: int = Form(350),
    pages: str = Form(""),
    lang: str = Form("ru"),
    tables: bool = Form(True),
    debug: bool = Form(False)
):
    try:
        import time
        t_start = time.time()
        logger.info("[pipeline] file received: name=%s content_type=%s",
                    file.filename, file.content_type)
        ct = (file.content_type or '').lower()
        if not any(t in ct for t in ['pdf', 'png', 'jpg', 'jpeg', 'image']):
            raise HTTPException(
                status_code=400, detail='Unsupported file type. Only PDF, PNG, JPG.')
        data = await file.read()
        logger.info("[pipeline] file size=%d bytes", len(data))
        if 'pdf' in ct:
            t_pdf0 = time.time()
            images = pdf_to_images(data, dpi=dpi)
            total = len(images)
            logger.info("[pipeline] pdf rendered pages=%d dpi=%d render_ms=%.1f",
                        total, dpi, (time.time()-t_pdf0)*1000)
            page_nums = parse_pages(pages, total)
            selected = [(images[p-1], p) for p in page_nums if 1 <= p <= total]
        else:
            img = Image.open(io.BytesIO(data))
            selected = [(img, 1)]
        # normalize / restrict language token (basic safety)
        lang_norm = ''.join(
            [c for c in lang.lower() if c.isalnum() or c in {'_', '-'}]) or 'ru'
        pages_out = []
        progress: List[Dict[str, Any]] = []
        for i, (img, idx) in enumerate(selected, 1):
            t_p0 = time.time()
            page_obj = process_image(img, idx, lang_norm, tables, debug)
            pages_out.append(page_obj)
            progress.append({
                "page": idx,
                "position": i,
                "total": len(selected),
                "lines": page_obj['text']['lines_count'],
                "chars": page_obj['text']['length'],
                "tables": len(page_obj.get('tables') or []),
                "time_ms": round((time.time()-t_p0)*1000, 2)
            })
        total_ms = (time.time()-t_start)*1000
        meta = {
            "pages_requested": len(selected),
            "total_ms": round(total_ms, 2),
        }
        return JSONResponse(content={"lang": lang_norm, "pages": pages_out, "progress": progress, "meta": meta})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
