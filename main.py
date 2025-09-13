import io
import os
from typing import List, Dict, Any
from functools import lru_cache
from PIL import Image
import pdf2image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import inspect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PaddleOCR API",
    description="CPU-only OCR service using PaddleOCR for text and table extraction",
    version="1.0.0"
)


def _filter_kwargs(callable_obj, desired: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
        accepted = {k: v for k, v in desired.items() if k in sig.parameters}
        dropped = set(desired.keys()) - set(accepted.keys())
        if dropped:
            logger.info(
                f"Dropped unsupported args for {callable_obj.__name__}: {dropped}")
        return accepted
    except (ValueError, TypeError):
        return desired


def create_ocr() -> PaddleOCR:
    desired = dict(use_angle_cls=True, lang='ru',
                   show_log=False, use_gpu=False)
    filtered = _filter_kwargs(PaddleOCR, desired)
    try:
        inst = PaddleOCR(**filtered)
        logger.info(f"Initialized PaddleOCR with args: {filtered}")
        return inst
    except Exception as e:
        logger.error(f"OCR init failed with {filtered}: {e}; retrying minimal")
        minimal = _filter_kwargs(PaddleOCR, dict(lang='ru'))
        inst = PaddleOCR(**minimal)
        logger.info(f"Initialized PaddleOCR with minimal args: {minimal}")
        return inst


@lru_cache(maxsize=1)
def get_ocr() -> PaddleOCR:
    return create_ocr()


ENABLE_TABLES = os.getenv("ENABLE_TABLES", "0").lower() in {"1", "true", "yes"}
_table_engine = None  # cached instance


def get_table_engine():
    global _table_engine
    if not ENABLE_TABLES:
        return None
    if _table_engine is not None:
        return _table_engine
    try:
        from paddleocr import PPStructure  # heavy import only if needed
    except ImportError:
        logger.warning("PP-Structure not available; table extraction disabled")
        return None
    desired = dict(lang='ru', show_log=False, use_gpu=False)
    filtered = _filter_kwargs(PPStructure, desired)
    try:
        engine = PPStructure(**filtered)
        logger.info(f"Initialized PPStructure with args: {filtered}")
        _table_engine = engine
        return _table_engine
    except Exception as e:
        logger.error(
            f"PPStructure init failed with {filtered}: {e}; retrying minimal")
        minimal = _filter_kwargs(PPStructure, dict(lang='ru'))
        engine = PPStructure(**minimal)
        logger.info(f"Initialized PPStructure with minimal args: {minimal}")
        _table_engine = engine
        return _table_engine


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
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi, fmt='RGB')
        return images
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error processing PDF: {e}")


def extract_text_from_image(image: Image.Image) -> str:
    try:
        img_array = io.BytesIO()
        image.save(img_array, format='PNG')
        img_array.seek(0)
        result = get_ocr().ocr(img_array.getvalue(), cls=True)
        lines: List[str] = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    lines.append(line[1][0])
        return '\n'.join(lines)
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""


def extract_tables_from_image(image: Image.Image) -> List[Dict[str, Any]]:
    engine = get_table_engine()
    if not engine:
        return []
    try:
        img_array = io.BytesIO()
        image.save(img_array, format='PNG')
        img_array.seek(0)
        result = engine(img_array.getvalue())
        tables: List[Dict[str, Any]] = []
        if result:
            for item in result:
                if item.get('type') == 'table' and 'res' in item:
                    table_data = item['res']
                    rows = []
                    if 'html' in table_data:
                        rows.append({"html": table_data['html']})
                    tables.append({
                        "rows": rows,
                        "bbox": item.get('bbox', [])
                    })
        return tables
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        return []


def process_image(image: Image.Image, page_index: int, return_html: bool = False) -> Dict[str, Any]:
    text_full = extract_text_from_image(image)
    tables = extract_tables_from_image(image)
    page: Dict[str, Any] = {
        "index": page_index,
        "text": {"full": text_full},
        "tables": tables
    }
    if return_html:
        html_content = f"<div class='page' data-page='{page_index}'>"
        html_content += f"<div class='text'>{text_full.replace(chr(10), '<br>')}</div>"
        for i, table in enumerate(tables):
            html_content += f"<div class='table' data-table='{i}'>"
            for row in table["rows"]:
                if "html" in row:
                    html_content += row["html"]
            html_content += "</div>"
        html_content += "</div>"
        page["html"] = html_content
    return page


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "PaddleOCR-API"}


@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    dpi: int = Form(350),
    pages: str = Form(""),
    return_html: bool = Form(False)
):
    try:
        content_type = file.content_type or ""
        if not any(t in content_type.lower() for t in ["pdf", "png", "jpg", "jpeg", "image"]):
            raise HTTPException(
                status_code=400, detail="Unsupported file type. Only PDF, PNG, and JPG supported.")
        data = await file.read()
        if 'pdf' in content_type.lower():
            images = pdf_to_images(data, dpi=dpi)
            total = len(images)
            page_numbers = parse_pages(pages, total)
            processed = [(images[p-1], p)
                         for p in page_numbers if 1 <= p <= total]
        else:
            image = Image.open(io.BytesIO(data))
            processed = [(image, 1)]
        pages_out = [process_image(img, idx, return_html)
                     for img, idx in processed]
        return JSONResponse(content={"lang": "ru", "pages": pages_out})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "PaddleOCR-API"}


@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    dpi: int = Form(350),
    pages: str = Form(""),
    return_html: bool = Form(False)
):
    """
    Analyze document (PDF/PNG/JPG) using OCR.

    Args:
        file: Document file (PDF, PNG, or JPG)
        dpi: DPI for PDF conversion (default: 350)
        pages: Page specification like "1-3,5" (default: all pages)
        return_html: Whether to include HTML representation

    Returns:
        JSON with OCR results including text and tables
    """
    try:
        # Validate file type
        content_type = file.content_type
        if not content_type or not any(
            ct in content_type.lower()
            for ct in ['pdf', 'png', 'jpg', 'jpeg', 'image']
        ):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only PDF, PNG, and JPG files are supported."
            )

        # Read file content
        file_content = await file.read()

        # Process based on file type
        if 'pdf' in content_type.lower():
            # Convert PDF to images
            images = pdf_to_images(file_content, dpi=dpi)
            total_pages = len(images)

            # Parse page specification
            page_numbers = parse_pages(pages, total_pages)

            # Process specified pages
            processed_images = []
            for page_num in page_numbers:
                if 1 <= page_num <= total_pages:
                    processed_images.append((images[page_num - 1], page_num))
        else:
            # Handle image files
            image = Image.open(io.BytesIO(file_content))
            processed_images = [(image, 1)]

        # Process each image
        pages_results = []
        for image, page_index in processed_images:
            page_result = process_image(image, page_index, return_html)
            pages_results.append(page_result)

        # Prepare response
        response = {
            "lang": "ru",
            "pages": pages_results
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
