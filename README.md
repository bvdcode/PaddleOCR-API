# PaddleOCR-API

A CPU-only OCR service built with FastAPI and PaddleOCR for text and table extraction from PDF, PNG, and JPG documents.

## Features

- **Text Recognition**: Extract text from documents using PaddleOCR with Russian language support
- **Table Detection**: Extract structured tables using PP-Structure
- **Multiple Formats**: Support for PDF, PNG, and JPG files
- **Flexible Page Selection**: Process specific pages with syntax like "1-3,5"
- **Configurable DPI**: Adjust PDF conversion quality
- **HTML Output**: Optional HTML representation of extracted content
- **Health Check**: Monitoring endpoint for service health

## API Endpoints

### GET /health

Health check endpoint that returns service status.

**Response:**

```json
{
  "status": "healthy",
  "service": "PaddleOCR-API"
}
```

### POST /analyze

Analyze documents for text and table extraction.

**Parameters:**

- `file` (required): Document file (PDF, PNG, or JPG)
- `dpi` (optional, default: 350): DPI for PDF conversion
- `pages` (optional, default: all): Page specification like "1-3,5"
- `return_html` (optional, default: false): Include HTML representation

**Response:**

```json
{
  "lang": "ru",
  "pages": [
    {
      "index": 1,
      "text": {
        "full": "Extracted text content..."
      },
      "tables": [
        {
          "rows": [
            {"html": "<table>...</table>"}
          ],
          "bbox": [x1, y1, x2, y2]
        }
      ],
      "html": "<div class='page'>...</div>"
    }
  ]
}
```

## Installation & Usage

### Quick Start with Prebuilt Docker Image (Recommended)

**No need to build locally!** Use the ready-to-go image:

```bash
docker run -p 8000:8000 bvdcode/paddleocrapi:v1.0
```

The service will be available at `http://localhost:8000`

### Using Docker Compose

1. Clone the repository (optional, only if you want to use docker-compose):

```bash
git clone https://github.com/bvdcode/PaddleOCR-API.git
cd PaddleOCR-API
```

2. Edit `docker-compose.yml` to use the prebuilt image:

```yaml
services:
  paddleocr-api:
    image: bvdcode/paddleocrapi:v1.0
    ports:
      - "8000:8000"
```

3. Start the service:

```bash
docker-compose up
```

---

### Local Build (for development)

1. Build the image manually:

```bash
docker build -t paddleocr-api .
```

2. Run the container:

```bash
docker run -p 8000:8000 paddleocr-api
```

### Local Development

1. Install Python 3.12 and system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install poppler-utils libgl1

# macOS
brew install poppler
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI
- **PaddleOCR**: OCR toolkit with Russian language support
- **PP-Structure**: Table detection and structure analysis
- **pdf2image**: PDF to image conversion
- **Pillow**: Image processing library
- **python-multipart**: File upload support

## Usage Examples

### Health Check

```bash
curl http://localhost:8000/health
```

### Analyze PDF Document

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@document.pdf" \
  -F "dpi=350" \
  -F "pages=1-3,5" \
  -F "return_html=true"
```

### Analyze Image

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@image.png"
```

## Configuration

The service can be configured through environment variables:

- `PYTHONUNBUFFERED=1`: Ensure Python output is not buffered
- `PYTHONIOENCODING=utf-8`: Set UTF-8 encoding for Python I/O

## System Requirements

- **CPU**: Multi-core processor recommended
- **Memory**: Minimum 2GB RAM, 4GB+ recommended
- **Storage**: At least 2GB free space for models and temporary files
- **OS**: Linux, macOS, or Windows with Docker

## Performance Notes

- First request may take longer as PaddleOCR downloads required models
- PDF processing time depends on document size and page count
- CPU-only processing is slower than GPU but more compatible
- Consider using page selection for large documents to improve performance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please create an issue in the GitHub repository.
