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
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PaddleOCR API",
              description="CPU-only OCR service", version="1.1.2")

# A pragmatic (not exhaustive) set of commonly used language codes that PaddleOCR accepts.
# Full list in PaddleOCR docs is large; models will auto-download on first use for a given lang code.
COMMON_LANGS = {
    'ch', 'en', 'fr', 'german', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'latin', 'arabic',
    'cyrillic', 'devanagari', 'ru', 'it', 'es', 'pt', 'hi', 'uk', 'tr', 'ug', 'fa', 'ur', 'bn', 'vi',
    'my', 'th', 'mr', 'ne', 'uz', 'kk', 'mn', 'he'  # etc.
}

# Optional restriction: if you want to limit which languages can be requested, set env OCR_LANG_WHITELIST="en,ru,fr".


def _get_lang_whitelist() -> Optional[set]:
    wl = os.getenv('OCR_LANG_WHITELIST')
    if not wl:
        return None
    return {x.strip().lower() for x in wl.split(',') if x.strip()}


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


def _sanitize_raw(obj: Any, depth: int = 0, max_depth: int = 4, max_list: int = 60) -> Any:
    """Best-effort conversion of Paddle raw output to JSON-safe structure.

    Truncates deeply nested / very long lists; converts numpy arrays to:
      {"shape": [...], "dtype": str, "sample": "truncated str(...)"}
    for large arrays, or full tolist() if tiny (<200 scalar values).
    """
    try:
        import numpy as _np
        if depth > max_depth:
            return '...'
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (list, tuple)):
            out = []
            for i, v in enumerate(obj):
                if i >= max_list:
                    out.append('...')
                    break
                out.append(_sanitize_raw(v, depth+1, max_depth, max_list))
            return out
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in list(obj.items())[:200]:  # cap huge dicts
                out[str(k)] = _sanitize_raw(v, depth+1, max_depth, max_list)
            if len(obj) > 200:
                out['...extra_keys'] = len(obj) - 200
            return out
        if hasattr(obj, 'tolist') and isinstance(obj, _np.ndarray):
            size = obj.size
            if size <= 200:
                return obj.tolist()
            return {
                'shape': list(obj.shape),
                'dtype': str(obj.dtype),
                'sample': str(obj.reshape(-1)[:16].tolist())
            }
        # Fallback generic string repr truncated
        rep = str(obj)
        if len(rep) > 300:
            rep = rep[:300] + '...'
        return rep
    except Exception as e:
        return f"<sanitize_error: {e}>"


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
        # Optional downscale if image side exceeds limit (helps avoid internal max_side_limit aborts)
        max_side_env = os.getenv('OCR_MAX_SIDE')
        max_side = None
        try:
            if max_side_env:
                max_side = int(max_side_env)
        except Exception:
            max_side = None
        if max_side and max(arr.shape[0], arr.shape[1]) > max_side:
            scale = max_side / float(max(arr.shape[0], arr.shape[1]))
            if scale < 1.0:
                new_w = int(arr.shape[1] * scale)
                new_h = int(arr.shape[0] * scale)
                try:
                    from PIL import Image as _Img
                    img_resized = Image.fromarray(arr).resize((new_w, new_h))
                    arr = np.array(img_resized)
                    logger.info(
                        f"[downscale] resized page to {new_w}x{new_h} (scale={scale:.3f}) due to OCR_MAX_SIDE={max_side}")
                except Exception as _re:
                    logger.warning(f"[downscale] failed resizing: {_re}")
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

        # Secondary: list of dicts
        if not structured and isinstance(result, list):
            # Case: each element dict may have its own dt_polys/rec_texts
            if all(isinstance(item, dict) for item in result):
                collected_any = False
                for item in result:
                    if all(k in item for k in ('dt_polys', 'rec_texts')):
                        polys = item.get('dt_polys') or []
                        texts = item.get('rec_texts') or []
                        scores = item.get('rec_scores') or []
                        for i, txt in enumerate(texts):
                            if not isinstance(txt, str) or not txt.strip():
                                continue
                            poly = polys[i] if i < len(polys) else None
                            if hasattr(poly, 'tolist'):
                                poly = poly.tolist()
                            score = None
                            if i < len(scores) and isinstance(scores[i], (int, float)):
                                score = float(scores[i])
                            structured.append(
                                {'text': txt, 'confidence': score, 'box': poly})
                            lines_plain.append(txt)
                            collected_any = True
                if not collected_any:
                    for item in result:
                        if isinstance(item, dict) and 'text' in item and isinstance(item['text'], str):
                            structured.append({
                                "text": item['text'],
                                "confidence": item.get('confidence') or item.get('score'),
                                "box": item.get('box') or item.get('bbox')
                            })
                            lines_plain.append(item['text'])

        # Tertiary: dict with 'data' or nested lists or modern keys (dt_polys/rec_texts)
        if not structured and isinstance(result, dict):
            # Modern batched inference style
            if all(k in result for k in ('dt_polys', 'rec_texts')):
                polys = result.get('dt_polys') or []
                texts = result.get('rec_texts') or []
                scores = result.get('rec_scores') or []
                for i, txt in enumerate(texts):
                    if not isinstance(txt, str) or not txt.strip():
                        continue
                    poly = polys[i] if i < len(polys) else None
                    score = None
                    if i < len(scores) and isinstance(scores[i], (int, float)):
                        score = float(scores[i])
                    # Convert numpy arrays to list
                    if hasattr(poly, 'tolist'):
                        poly = poly.tolist()
                    structured.append({
                        'text': txt,
                        'confidence': score,
                        'box': poly
                    })
                    lines_plain.append(txt)
            if not structured:
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

        # Deep recursive fallback (guarded) if still empty (no geometry yet)
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
            # Ensure no numpy arrays leak (JSON safe)
            if isinstance(line.get('box'), np.ndarray):
                line['box'] = line['box'].tolist()

        # Manual detector + recognizer fallback if still no boxes at all
        if structured and not any('bbox' in ln for ln in structured):
            try:
                det_model = getattr(ocr_engine, 'text_detector', None)
                rec_model = getattr(ocr_engine, 'text_recognizer', None)
                if det_model and rec_model:
                    det_res = det_model(arr)
                    # det_res expected: list of polys
                    polys = det_res if isinstance(det_res, list) else []
                    rec_inputs = []
                    boxes_norm = []
                    from PIL import Image as _Img
                    h, w = arr.shape[0], arr.shape[1]
                    for poly in polys:
                        if hasattr(poly, 'tolist'):
                            poly_l = poly.tolist()
                        else:
                            poly_l = poly
                        bbox = _coerce_bbox(poly_l)
                        if not bbox:
                            continue
                        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
                        x2 = min(w, x2)
                        y2 = min(h, y2)
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            continue
                        crop = arr[y1:y2, x1:x2]
                        rec_inputs.append(crop)
                        boxes_norm.append(poly_l)
                    if rec_inputs:
                        rec_res = rec_model(rec_inputs)
                        # rec_res: list of (text, score)
                        new_struct = []
                        for (txt, score), poly in zip(rec_res, boxes_norm):
                            if not isinstance(txt, str) or not txt.strip():
                                continue
                            new_struct.append({'text': txt, 'confidence': float(
                                score) if isinstance(score, (int, float)) else None, 'box': poly})
                        if new_struct:
                            structured = new_struct
                            lines_plain = [ln['text'] for ln in structured]
                            for ln in structured:
                                bbox = _coerce_bbox(ln.get('box'))
                                if bbox:
                                    ln['bbox'] = bbox
            except Exception as _e:
                logger.info(
                    f"manual detector-recognizer fallback failed: {_e}")
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
        sanitized = _sanitize_raw(result)
        return {"full_text": full_text, "lines": structured, "raw": result if debug else None, "raw_truncated": raw_truncated, "raw_full": sanitized, "stats": stats}
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
        sanitized = _sanitize_raw(result)
        return {"tables": tables, "raw": result if debug else None, "raw_summary": raw_summary, "raw_truncated": raw_truncated, "raw_full": sanitized}
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        return {"tables": [], "raw": None, "raw_truncated": None, "error": str(e)}


def process_image(image: Image.Image, page_index: int, lang: str, do_tables: bool, debug: bool) -> Dict[str, Any]:
    """Run OCR + (optional) table detection for a single page.

    HTML output was removed per user request; only structured JSON is returned.
    """
    import time
    t0 = time.time()
    try:
        text_struct = extract_text_from_image(image, lang)
    except Exception as e:
        logger.error(
            f"[page {page_index}] OCR fatal error: {type(e).__name__}: {e}")
        text_struct = {"full_text": "", "lines": [], "error": str(e)}
    ocr_ms = (time.time() - t0)*1000
    t1 = time.time()
    table_struct: Dict[str, Any] = {"tables": []}
    if do_tables:
        try:
            table_struct = extract_tables_from_image(image, debug)
        except Exception as e:
            logger.error(
                f"[page {page_index}] table fatal error: {type(e).__name__}: {e}")
            table_struct = {"tables": [], "error": str(e)}
    table_ms = (time.time() - t1)*1000 if do_tables else 0.0
    text_full = text_struct["full_text"]
    stats = text_struct.get('stats') or {}
    # Adapt geometry representation: convert box (polygon) -> list[{x,y}], bbox -> {x,y,w,h}

    def poly_to_points(poly):
        if not isinstance(poly, (list, tuple)):
            return None
        pts = []
        for p in poly:
            if isinstance(p, (list, tuple)) and len(p) >= 2 and all(isinstance(v, (int, float)) for v in p[:2]):
                pts.append({'x': float(p[0]), 'y': float(p[1])})
        return pts if pts else None

    def bbox_to_obj(b):
        if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            x1, y1, x2, y2 = b
            return {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}
        return None
    # Deep copy/transform lines
    transformed_lines: List[Dict[str, Any]] = []
    for ln in text_struct['lines']:
        if not isinstance(ln, dict):
            continue
        new_ln = dict(ln)
        if 'box' in new_ln:
            pts = poly_to_points(new_ln['box'])
            if pts is not None:
                new_ln['box'] = pts
            else:
                # drop invalid
                new_ln.pop('box', None)
        if 'bbox' in new_ln:
            bb = bbox_to_obj(new_ln['bbox'])
            if bb is not None:
                new_ln['bbox'] = bb
            else:
                new_ln.pop('bbox', None)
        transformed_lines.append(new_ln)
    # Transform tables cells geometry similarly
    transformed_tables = []
    for tb in table_struct.get('tables', []) or []:
        if not isinstance(tb, dict):
            continue
        new_tb = dict(tb)
        # table bbox may be list of 4 numbers
        if 'bbox' in new_tb:
            bb = bbox_to_obj(new_tb['bbox'])
            if bb is not None:
                new_tb['bbox'] = bb
        cells_t = []
        for c in new_tb.get('cells', []) or []:
            if not isinstance(c, dict):
                continue
            nc = dict(c)
            if 'bbox' in nc:
                bb = bbox_to_obj(nc['bbox'])
                if bb is not None:
                    nc['bbox'] = bb
            cells_t.append(nc)
        new_tb['cells'] = cells_t
        transformed_tables.append(new_tb)
    page: Dict[str, Any] = {
        "index": page_index,
        "text": {
            "full": text_full,
            "length": len(text_full),
            "lines": transformed_lines,
            "lines_count": len(transformed_lines)
        },
        "tables": transformed_tables,
        "metrics": {
            "ocr_ms": round(ocr_ms, 2),
            "table_ms": round(table_ms, 2),
            "total_ms": round((time.time() - t0)*1000, 2),
            **({k: round(v, 6) for k, v in stats.items()} if stats else {})
        }
    }
    if text_struct.get('error'):
        page['text']['error'] = text_struct['error']
    if table_struct.get('error'):
        page.setdefault('tables_meta', {})['error'] = table_struct['error']
    # Secondary heuristic: if no tables detected but many numeric-like lines that appear columnar, attempt simple column grouping.
    if do_tables and not page['tables']:
        lines_with_bbox = [ln for ln in page['text']['lines'] if isinstance(
            ln, dict) and ln.get('bbox') and isinstance(ln.get('text'), str)]
        if len(lines_with_bbox) >= 12:
            # Cluster by y into rows
            rows: List[List[Dict[str, Any]]] = []
            for ln in sorted(lines_with_bbox, key=lambda x: x['bbox'][1]):
                placed = False
                for r in rows:
                    y_min = max(ln['bbox'][1], min(c['bbox'][1] for c in r))
                    y_max = min(ln['bbox'][3], max(c['bbox'][3] for c in r))
                    h = min(ln['bbox'][3]-ln['bbox'][1], max(c['bbox'][3]
                            for c in r)-min(c['bbox'][1] for c in r))
                    if h > 0 and (y_max - y_min)/h >= 0.4:
                        r.append(ln)
                        placed = True
                        break
                if not placed:
                    rows.append([ln])
            # Determine if there are multiple rows with 2+ columns -> treat as table
            multi = [r for r in rows if len(r) >= 2]
            if len(multi) >= 3 and sum(len(r) for r in multi) >= 10:
                all_cells = []
                x1 = min(min(c['bbox'][0] for c in r) for r in multi)
                y1 = min(min(c['bbox'][1] for c in r) for r in multi)
                x2 = max(max(c['bbox'][2] for c in r) for r in multi)
                y2 = max(max(c['bbox'][3] for c in r) for r in multi)
                for r in multi:
                    for c in r:
                        all_cells.append(
                            {'text': c['text'], 'bbox': c['bbox']})
                inferred_table = {'bbox': [x1, y1, x2, y2], 'cells': all_cells, 'cells_count': len(
                    all_cells), 'inferred_from': 'ocr_lines'}
                page['tables'].append(inferred_table)
    page["debug"] = {
        "ocr_raw_present": bool(text_struct.get("raw")),
        "tables_raw_present": bool(table_struct.get("raw")),
        "tables_raw_count": len(table_struct.get("raw", []) if isinstance(table_struct.get("raw"), list) else []),
    }
    return page


@app.get('/health')
async def health():
    return {"status": "healthy", "service": "PaddleOCR-API"}


@app.get('/languages')
async def list_languages():
    """Return available common languages and (optional) enforced whitelist.

    This does NOT guarantee the model is already downloaded; PaddleOCR will fetch
    the needed weights on first use. The list is intentionally trimmed – Paddle's
    full universe is larger. Add or remove as desired.
    """
    wl = _get_lang_whitelist()
    return {
        "common": sorted(COMMON_LANGS),
        "whitelist": sorted(wl) if wl else None,
        "note": "Pass ?refresh=1 to force logic change in future versions (noop now)."
    }


def _validate_lang(lang: Optional[str]) -> str:
    """Normalize and validate language code.

    Rules:
      - Must be provided and not empty after normalization.
      - If OCR_LANG_WHITELIST set -> must be in whitelist.
      - Else if OCR_LANG_ALLOW_ANY in {1,true,yes} -> accept anything (best-effort; Paddle may fail later).
      - Else must be in COMMON_LANGS set; otherwise 400.
    """
    if not lang or not lang.strip():
        raise HTTPException(
            status_code=400, detail="Parameter 'lang' is required and must be non-empty.")
    lang_norm = ''.join(
        [c for c in lang.lower() if c.isalnum() or c in {'_', '-'}])
    if not lang_norm:
        raise HTTPException(
            status_code=400, detail="Parameter 'lang' after normalization became empty.")
    wl = _get_lang_whitelist()
    allow_any = os.getenv('OCR_LANG_ALLOW_ANY', '0').lower() in {
        '1', 'true', 'yes'}
    if wl is not None:
        if lang_norm not in wl:
            raise HTTPException(
                status_code=400, detail=f"Language '{lang_norm}' not in whitelist. Allowed: {sorted(wl)}")
    else:
        if not allow_any and lang_norm not in COMMON_LANGS:
            raise HTTPException(
                status_code=400, detail=f"Language '{lang_norm}' not recognized. Allowed common set: {sorted(COMMON_LANGS)}. Set OCR_LANG_ALLOW_ANY=1 to bypass.")
        if allow_any and lang_norm not in COMMON_LANGS:
            logger.info(
                f"[lang] '{lang_norm}' not in COMMON_LANGS; proceeding due to OCR_LANG_ALLOW_ANY.")
    return lang_norm


@app.post('/analyze')
async def analyze(
    file: UploadFile = File(...),
    dpi: int = Form(350),
    pages: str = Form(""),
    lang: str = Form(..., description="Explicit PaddleOCR language code (required). Use /languages for a list."),
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
        # Validate & normalize language (required; no fallback default)
        lang_norm = _validate_lang(lang)

        # Base document hash (independent of requested page subset). Not including pages param here for reusability.
        base_h = hashlib.sha256(); base_h.update(data)
        base_meta_part = f"|lang={lang_norm}|tables={tables}|dpi={dpi}".encode('utf-8')
        base_h.update(base_meta_part); base_hash = base_h.hexdigest()
        cache_dir = os.path.join(os.getcwd(), 'cache', base_hash)
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass
        # Manifest file summarizing already processed pages
        manifest_path = os.path.join(cache_dir, 'manifest.json')
        manifest = {"base_hash": base_hash, "pages": {}, "version": "1"}
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as mf:
                    mdata = json.load(mf)
                    if isinstance(mdata, dict) and mdata.get('base_hash') == base_hash:
                        manifest = mdata
            except Exception as e:
                logger.warning(f"[cache] manifest load failed: {e}")
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
    # (lang already validated and normalized above)
        pages_out: List[Dict[str, Any]] = []
        progress: List[Dict[str, Any]] = []
        failed_pages = 0
        total_pages_selected = len(selected)
        for i, (img, idx) in enumerate(selected, 1):
            t_p0 = time.time()
            page_key = str(idx)
            page_cache_name = f"page_{idx}.json"  # per-page snapshot
            page_cache_path = os.path.join(cache_dir, page_cache_name)
            page_obj: Dict[str, Any]
            cache_hit = False
            if os.path.isfile(page_cache_path):
                try:
                    with open(page_cache_path, 'r', encoding='utf-8') as pf:
                        cached_page = json.load(pf)
                    # minimal validation
                    if isinstance(cached_page, dict) and cached_page.get('index') == idx:
                        page_obj = cached_page
                        cache_hit = True
                        logger.info(f"[cache:page] hit base={base_hash} page={idx}")
                    else:
                        page_obj = process_image(img, idx, lang_norm, tables, debug)
                except Exception as e:
                    logger.warning(f"[cache:page] read failed page={idx}: {e}; recompute")
                    page_obj = process_image(img, idx, lang_norm, tables, debug)
            else:
                page_obj = process_image(img, idx, lang_norm, tables, debug)
                # Write page cache
                try:
                    with open(page_cache_path, 'w', encoding='utf-8') as pf:
                        json.dump(page_obj, pf, ensure_ascii=False, separators=(',', ':'), indent=None)
                except Exception as e:
                    logger.warning(f"[cache:page] write failed page={idx}: {e}")
            # Update manifest entry
            manifest['pages'][page_key] = {
                'cache_file': page_cache_name,
                'cached': True,
                'error': page_obj.get('text', {}).get('error') or page_obj.get('tables_meta', {}).get('error') if page_obj.get('tables_meta') else None
            }
            if page_obj.get('text', {}).get('error'):
                failed_pages += 1
            pages_out.append(page_obj)
            progress.append({
                'page': idx,
                'position': i,
                'total': total_pages_selected,
                'lines': page_obj.get('text', {}).get('lines_count', 0),
                'chars': page_obj.get('text', {}).get('length', 0),
                'tables': len(page_obj.get('tables') or []),
                'time_ms': round((time.time() - t_p0)*1000, 2),
                'cache': cache_hit,
                'error': page_obj.get('text', {}).get('error') or page_obj.get('tables_meta', {}).get('error') if page_obj.get('tables_meta') else None
            })
        # Persist manifest
        try:
            with open(manifest_path, 'w', encoding='utf-8') as mf:
                json.dump(manifest, mf, ensure_ascii=False, separators=(',', ':'), indent=None)
        except Exception as e:
            logger.warning(f"[cache] manifest write failed: {e}")
        total_ms = (time.time()-t_start)*1000
        meta = {
            "pages_requested": len(selected),
            "total_ms": round(total_ms, 2),
            "pages_failed": failed_pages,
        }
        response_obj = {"lang": lang_norm, "pages": pages_out,
                        "progress": progress, "meta": meta, "hash": base_hash}
        # No whole-document cache write now; per-page + manifest only
        return JSONResponse(content=response_obj)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Document-level failure: {type(e).__name__}: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
