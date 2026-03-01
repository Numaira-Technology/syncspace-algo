from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from tempfile import NamedTemporaryFile
from main import main as process_documents
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing API",
    description="API for processing and updating documents based on Excel data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define response models
class Change(BaseModel):
    original_text: str
    modified_text: str
    confidence: float

class ProcessingResult(BaseModel):
    number_of_changes: int
    results: List[Change]
    output_file_path: str

class SuccessResponse(BaseModel):
    status: str = "success"
    data: ProcessingResult

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str

@app.post("/run-syncspace/", 
    response_model=SuccessResponse,
    responses={500: {"model": ErrorResponse}}
)
async def process_documents_endpoint(
    docx_file: UploadFile = File(...),
    excel_file: UploadFile = File(...)
):
    tmp_docx_path = None
    tmp_excel_path = None
    try:
        # Save uploaded DOCX file to a temporary file
        with NamedTemporaryFile(delete=False, suffix='.docx') as tmp_docx:
            shutil.copyfileobj(docx_file.file, tmp_docx)
            tmp_docx_path = tmp_docx.name
            
        # Save uploaded Excel file to a temporary file
        with NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
            shutil.copyfileobj(excel_file.file, tmp_excel)
            tmp_excel_path = tmp_excel.name

        # Process the documents
        result = process_documents(tmp_docx_path, tmp_excel_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
    finally:
        # Ensure temporary files are removed even if an error occurs
        if tmp_docx_path and os.path.exists(tmp_docx_path):
            os.unlink(tmp_docx_path)
        if tmp_excel_path and os.path.exists(tmp_excel_path):
            os.unlink(tmp_excel_path)

class JsonSyncspaceRequest(BaseModel):
    docx_data: Dict[str, Any]
    excel_data: Dict[str, Any]


@app.post("/run-syncspace-json/",
    responses={500: {"model": ErrorResponse}}
)
async def process_documents_json_endpoint(request: JsonSyncspaceRequest):
    """
    Process pre-extracted document data directly without file uploads.

    Accepts:
        docx_data: { "data": { "sentence1": "...", "sentence2": "..." } }
        excel_data: { "data": [ { "row_header": "...", "values": { "year": value } }, ... ] }

    Returns the same response format as /run-syncspace/.
    """
    try:
        from utils.document_processing.vision_processor import analyze_and_update_document_from_json

        word_sentences = request.docx_data.get("data", {})
        excel_rows = request.excel_data.get("data", [])

        logger.info(f"[json endpoint] word sentences: {len(word_sentences)}, excel rows: {len(excel_rows)}")

        results = analyze_and_update_document_from_json(word_sentences, excel_rows)

        return JSONResponse(content={
            "status": "success",
            "data": {
                "number_of_changes": len(results),
                "results": [
                    {
                        "original_text": orig,
                        "modified_text": mod,
                        "confidence": float(conf),
                        "description": desc
                    } for orig, mod, conf, desc in results
                ]
            }
        })

    except Exception as e:
        logger.error(f"Error in json endpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


class ImageSyncspaceRequest(BaseModel):
    docx_image: str   # base64 PNG of the rendered Word document
    excel_image: str  # base64 PNG of the rendered Excel sheet


@app.post("/run-syncspace-images/",
    responses={500: {"model": ErrorResponse}}
)
async def process_images_endpoint(request: ImageSyncspaceRequest):
    """
    Process Word and Excel files sent as base64 PNG images.

    Uses a vision LLM to OCR both images, extracts meaningful content and
    financial data, then runs the standard text-LLM update analysis.

    Accepts:
        docx_image: base64-encoded PNG of the Word document
        excel_image: base64-encoded PNG of the Excel sheet

    Returns the same response format as /run-syncspace/.
    """
    try:
        from utils.document_processing.vision_processor import analyze_and_update_from_images

        logger.info(
            f"[images endpoint] docx_image size: {len(request.docx_image)} chars, "
            f"excel_image size: {len(request.excel_image)} chars"
        )

        results = analyze_and_update_from_images(request.docx_image, request.excel_image)

        return JSONResponse(content={
            "status": "success",
            "data": {
                "number_of_changes": len(results),
                "results": [
                    {
                        "original_text": orig,
                        "modified_text": mod,
                        "confidence": float(conf),
                        "description": desc
                    } for orig, mod, conf, desc in results
                ]
            }
        })

    except Exception as e:
        logger.error(f"Error in images endpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
