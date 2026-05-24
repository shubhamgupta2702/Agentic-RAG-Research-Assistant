from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.graph.workflow import workflow
from src.rag.data_ingest import ingest_pdf
from src.logger.logger import logger
import tempfile
import os

MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    route: str

@router.post("/chat", response_model=AnswerResponse)
async def chat(request: QuestionRequest):
    try:
        logger.info(f"Received chat request: {request.question}")
        result = workflow.invoke({
            "question": request.question,
            "context": [],
            "sources": [],
            "messages": [],
        })
        return AnswerResponse(
            question=request.question,
            answer=result.get("generation", "No answer generated."),
            sources=result.get("sources", []),
            route=result.get("route", "unknown"),
        )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        await ingest_pdf(tmp_path)
        os.unlink(tmp_path)
        
        return {"message": f"Successfully ingested: {file.filename}"}
    except Exception as e:
        logger.error(f"Ingest error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health():
    return {
        "status": "ok",
        "version": MODEL_VERSION
    }