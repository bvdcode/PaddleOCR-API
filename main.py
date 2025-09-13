import io
import os
import re
from typing import List, Optional, Dict, Any
from PIL import Image
import pdf2image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import paddleocr
from paddleocr import PaddleOCR
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PaddleOCR API",
    description="CPU-only OCR service using PaddleOCR for text and table extraction",
    version="1.0.0"
)

# Initialize PaddleOCR (CPU version)
ocr = PaddleOCR(
    use_angle_cls=True,  # Enable angle classification
    lang='ru',           # Russian language
    use_gpu=False,       # CPU-only
    show_log=False       # Reduce log output
)

# Initialize PP-Structure for table detection
try:
    from paddleocr import PPStructure
    table_engine = PPStructure(
        lang='ru',
        use_gpu=False,
        show_log=False
    )
except ImportError:
    logger.warning("PP-Structure not available, table detection will be disabled")
    table_engine = None


def parse_pages(pages_str: str, total_pages: int) -> List[int]:
    """Parse page specification like '1-3,5' into list of page numbers."""
    if not pages_str:
        return list(range(1, total_pages + 1))
    
    pages = []
    for part in pages_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    
    # Filter valid page numbers
    return [p for p in pages if 1 <= p <= total_pages]


def pdf_to_images(pdf_bytes: bytes, dpi: int = 350) -> List[Image.Image]:
    """Convert PDF to list of PIL Images."""
    try:
        images = pdf2image.convert_from_bytes(
            pdf_bytes, 
            dpi=dpi,
            fmt='RGB'
        )
        return images
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")


def extract_text_from_image(image: Image.Image) -> str:
    """Extract text from image using PaddleOCR."""
    try:
        # Convert PIL Image to format expected by PaddleOCR
        img_array = io.BytesIO()
        image.save(img_array, format='PNG')
        img_array.seek(0)
        
        result = ocr.ocr(img_array.getvalue(), cls=True)
        
        # Extract text from results
        text_lines = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text_lines.append(line[1][0])
        
        return '\n'.join(text_lines)
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""


def extract_tables_from_image(image: Image.Image) -> List[Dict[str, Any]]:
    """Extract tables from image using PP-Structure."""
    if not table_engine:
        return []
    
    try:
        # Convert PIL Image to format expected by PP-Structure
        img_array = io.BytesIO()
        image.save(img_array, format='PNG')
        img_array.seek(0)
        
        result = table_engine(img_array.getvalue())
        
        tables = []
        if result:
            for item in result:
                if item.get('type') == 'table' and 'res' in item:
                    table_data = item['res']
                    # Convert table structure to rows format
                    rows = []
                    if 'html' in table_data:
                        # Parse HTML table structure (simplified)
                        html_content = table_data['html']
                        # For now, just store the HTML
                        rows.append({"html": html_content})
                    
                    tables.append({
                        "rows": rows,
                        "bbox": item.get('bbox', [])
                    })
        
        return tables
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        return []


def process_image(image: Image.Image, page_index: int, return_html: bool = False) -> Dict[str, Any]:
    """Process a single image and return OCR results."""
    # Extract text
    text_full = extract_text_from_image(image)
    
    # Extract tables
    tables = extract_tables_from_image(image)
    
    page_result = {
        "index": page_index,
        "text": {
            "full": text_full
        },
        "tables": tables
    }
    
    if return_html:
        # Generate basic HTML representation
        html_content = f"<div class='page' data-page='{page_index}'>"
        html_content += f"<div class='text'>{text_full.replace(chr(10), '<br>')}</div>"
        
        for i, table in enumerate(tables):
            html_content += f"<div class='table' data-table='{i}'>"
            for row in table["rows"]:
                if "html" in row:
                    html_content += row["html"]
            html_content += "</div>"
        
        html_content += "</div>"
        page_result["html"] = html_content
    
    return page_result


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
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)