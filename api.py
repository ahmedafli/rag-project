import os
import shutil
import uuid
from fastapi import UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from db import get_db, Document
from ingest import ingest_files
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import RAGPipeline
from typing import List

app = FastAPI(title="RAG Assistant API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# CORS: allows a frontend running on a different port/domain (e.g. Streamlit
# on :8501 or Next.js on :3000) to actually call this API from the browser.
# Without this, browsers block the request by default for security reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pipeline ONCE when the server starts, not per-request
pipeline = RAGPipeline()


class QuestionRequest(BaseModel):
    question: str
    k: int = 5


class SourceItem(BaseModel):
    n: int
    source: str
    rerank_score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    result = pipeline.answer(request.question, k=request.k)
    return result

@app.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    saved = []

    for file in files:
        if not file.filename.lower().endswith((".pdf", ".txt", ".md")):
            # Skip invalid files rather than failing the whole batch
            continue

        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        saved_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")

        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        doc = Document(id=doc_id, filename=file.filename, file_path=saved_path, status="processing")
        db.add(doc)
        saved.append((doc_id, saved_path))

    db.commit()  # commit all rows together, once

    # Queue one background task that processes all of them SEQUENTIALLY,
    # not N separate tasks firing in parallel -- avoids hammering the
    # rate-limited embedding API with concurrent requests.
    background_tasks.add_task(process_uploads_sequentially, saved)

    return [{"id": doc_id, "status": "processing"} for doc_id, _ in saved]