import io
import os
from typing import List, Dict, Any, Optional
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
    desired = dict(lang='en', show_log=False, use_gpu=False, table=True)
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
        return pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi, fmt='RGB')
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error processing PDF: {e}")


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

        # We'll collect structured line objects: {text, confidence, box}
        structured: List[Dict[str, Any]] = []
        lines_plain: List[str] = []

        # Primary (legacy) pattern
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, list):
                for entry in first:
                    # legacy entry: [box, (text, conf)]
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
                            structured.append({
                                "text": text_val,
                                "confidence": conf_val,
                                "box": box if isinstance(box, (list, tuple)) else None
                            })

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

        full_text = '\n'.join(lines_plain)
        if os.getenv('OCR_LOG_COUNTS', '1').lower() in {'1', 'true', 'yes'}:
            logger.info(
                f"OCR page: lines={len(structured)} chars={len(full_text)} time_ms={elapsed:.1f}")
        if debug and not structured:
            raw_repr = str(result)
            logger.debug(f"RAW OCR RESULT SAMPLE (trim): {raw_repr[:800]}")
        return {"full_text": full_text, "lines": structured}
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return {"full_text": "", "lines": []}


def extract_tables_from_image(image: Image.Image) -> List[Dict[str, Any]]:
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
        return []
    try:
        arr = np.array(image.convert('RGB'))
        result = engine(arr)
        tables: List[Dict[str, Any]] = []
        total_cells = 0
        if result:
            for item in result:
                if not (isinstance(item, dict) and item.get('type') == 'table'):
                    continue
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
                    bbox = c.get('bbox') or c.get(
                        'box') or c.get('poly') or None
                    cells_out.append({'text': txt, 'bbox': bbox})
                total_cells += len(cells_out)
                tables.append({'bbox': item.get('bbox', []),
                              'cells': cells_out, 'cells_count': len(cells_out)})
        if (os.getenv('TABLE_DEBUG', '0').lower() in {'1', 'true', 'yes'}) or (os.getenv('OCR_DEBUG', '0').lower() in {'1', 'true', 'yes'}):
            logger.info(
                f"table detection: raw_items={len(result) if isinstance(result, list) else 'n/a'} tables={len(tables)} total_cells={total_cells}")
        return tables
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        return []


def process_image(image: Image.Image, page_index: int, lang: str, do_tables: bool) -> Dict[str, Any]:
    """Run OCR + (optional) table detection for a single page.

    HTML output was removed per user request; only structured JSON is returned.
    """
    text_struct = extract_text_from_image(image, lang)
    text_full = text_struct["full_text"]
    tables = extract_tables_from_image(image) if do_tables else []
    page: Dict[str, Any] = {"index": page_index,
                            "text": {"full": text_full, "length": len(text_full), "lines": text_struct["lines"]}, "tables": tables}
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
    return_text: bool = Form(True),
    return_raw: bool = Form(False)
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
        for i, (img, idx) in enumerate(selected, 1):
            t_p0 = time.time()
            logger.info("[pipeline] page %d/%d start", i, len(selected))
            page_obj = process_image(img, idx, lang_norm, tables)
            pages_out.append(page_obj)
            logger.info("[pipeline] page %d/%d processed lines=%s chars=%s tables=%d time_ms=%.1f", i, len(selected),
                        len(page_obj['text'].get('lines', [])
                            ) if 'lines' in page_obj['text'] else 'n/a',
                        page_obj['text'].get('length'), len(page_obj.get('tables') or []), (time.time()-t_p0)*1000)
        logger.info("[pipeline] done pages=%d total_ms=%.1f",
                    len(selected), (time.time()-t_start)*1000)
        if not return_text:
            for p in pages_out:
                # keep only length & line count metadata
                p_text = p['text']
                p['text'] = {"length": p_text['length'],
                             "lines_count": len(p_text.get('lines') or [])}
        elif not return_raw:
            # remove per-line details if raw not requested
            for p in pages_out:
                p['text'].pop('lines', None)
        return JSONResponse(content={"lang": lang_norm, "pages": pages_out, "raw": return_raw})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
