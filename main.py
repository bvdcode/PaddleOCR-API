import io
import os
from typing import List, Dict, Any
from PIL import Image
import pdf2image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import inspect
import logging

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


ENABLE_TABLES = os.getenv("ENABLE_TABLES", "0").lower() in {"1", "true", "yes"}
_TABLE_ENGINE = None


def get_table_engine():
    global _TABLE_ENGINE
    if not ENABLE_TABLES:
        return None
    if _TABLE_ENGINE is not None:
        return _TABLE_ENGINE
    try:
        from paddleocr import PPStructure
    except ImportError:
        logger.warning("PP-Structure not available")
        return None
    desired = dict(lang='ru', show_log=False, use_gpu=False)
    filtered = _filter_kwargs(PPStructure, desired)
    try:
        engine = PPStructure(**filtered)
        logger.info(f"Initialized PPStructure with args: {filtered}")
        _TABLE_ENGINE = engine
        return _TABLE_ENGINE
    except Exception as e:
        logger.warning(f"PPStructure init failed: {e}; retry minimal")
        minimal = _filter_kwargs(PPStructure, dict(lang='ru'))
        engine = PPStructure(**minimal)
        logger.info(f"Initialized PPStructure with minimal args: {minimal}")
        _TABLE_ENGINE = engine
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


def extract_text_from_image(image: Image.Image, lang: str) -> str:
    try:
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        ocr_engine = get_ocr(lang)
        data = buf.getvalue()
        result = None
        # Prefer new predict API if available
        if hasattr(ocr_engine, 'predict'):
            try:
                result = ocr_engine.predict(images=[data])
            except TypeError:
                # some versions might accept raw bytes
                try:
                    result = ocr_engine.predict(data)
                except Exception:
                    result = None
            except Exception:
                result = None
        # Fallback to legacy .ocr()
        if result is None:
            try:
                result = ocr_engine.ocr(data, cls=True)
            except TypeError:
                result = ocr_engine.ocr(data)
        lines: List[str] = []
        try:
            # Attempt to normalize different result schemas
            if isinstance(result, list):
                # Typical legacy format: [ [ [box, (text, conf)], ... ] ]
                candidate = result[0] if result and isinstance(
                    result[0], list) else result
                for line in candidate:
                    if isinstance(line, (list, tuple)):
                        if len(line) >= 2:
                            txt_part = line[1]
                            if isinstance(txt_part, (list, tuple)) and txt_part:
                                lines.append(str(txt_part[0]))
                            elif isinstance(txt_part, str):
                                lines.append(txt_part)
                    elif isinstance(line, dict) and 'text' in line:
                        lines.append(str(line['text']))
            elif isinstance(result, dict):
                if 'data' in result and isinstance(result['data'], list):
                    for item in result['data']:
                        if isinstance(item, dict) and 'text' in item:
                            lines.append(str(item['text']))
        except Exception as parse_err:
            logger.warning(f"Failed parsing OCR result: {parse_err}")
        return '\n'.join([l for l in lines if l])
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""


def extract_tables_from_image(image: Image.Image) -> List[Dict[str, Any]]:
    engine = get_table_engine()
    if not engine:
        return []
    try:
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        result = engine(buf.getvalue())
        tables: List[Dict[str, Any]] = []
        if result:
            for item in result:
                if item.get('type') == 'table' and 'res' in item:
                    table_data = item['res']
                    rows = []
                    if 'html' in table_data:
                        rows.append({'html': table_data['html']})
                    tables.append({'rows': rows, 'bbox': item.get('bbox', [])})
        return tables
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        return []


def process_image(image: Image.Image, page_index: int, lang: str, return_html: bool = False) -> Dict[str, Any]:
    text_full = extract_text_from_image(image, lang)
    tables = extract_tables_from_image(image)
    page: Dict[str, Any] = {"index": page_index,
                            "text": {"full": text_full, "length": len(text_full)}, "tables": tables}
    if return_html:
        html = f"<div class='page' data-page='{page_index}'>"
        html += f"<div class='text'>{text_full.replace(chr(10), '<br>')}</div>"
        for i, table in enumerate(tables):
            html += f"<div class='table' data-table='{i}'>"
            for row in table['rows']:
                if 'html' in row:
                    html += row['html']
            html += "</div>"
        html += "</div>"
        page['html'] = html
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
    return_html: bool = Form(False),
    return_text: bool = Form(True)
):
    try:
        ct = (file.content_type or '').lower()
        if not any(t in ct for t in ['pdf', 'png', 'jpg', 'jpeg', 'image']):
            raise HTTPException(
                status_code=400, detail='Unsupported file type. Only PDF, PNG, JPG.')
        data = await file.read()
        if 'pdf' in ct:
            images = pdf_to_images(data, dpi=dpi)
            total = len(images)
            page_nums = parse_pages(pages, total)
            selected = [(images[p-1], p) for p in page_nums if 1 <= p <= total]
        else:
            img = Image.open(io.BytesIO(data))
            selected = [(img, 1)]
        # normalize / restrict language token (basic safety)
        lang_norm = ''.join(
            [c for c in lang.lower() if c.isalnum() or c in {'_', '-'}]) or 'ru'
        pages_out = [process_image(img, idx, lang_norm, return_html)
                     for img, idx in selected]
        if not return_text:
            for p in pages_out:
                p['text'].pop('full', None)
        return JSONResponse(content={"lang": lang_norm, "pages": pages_out})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
