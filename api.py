import os
import pickle
import shutil
import uuid
from fastapi import UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from qdrant_client import models
from db import get_db, SessionLocal, Document
from ingest import ingest_files
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import RAGPipeline
from typing import List
import json
import config
from db import Conversation, Message

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
    source_filenames: list[str] = []
    session_id: str
    conversation_id: str | None = None

class SourceItem(BaseModel):
    n: int
    source: str
    doc_id: str | None = None
    page: int | str | None = None
    type: str | None = None
    text: str
    rerank_score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest, db: Session = Depends(get_db)):
    convo = None
    if request.conversation_id:
        convo = db.query(Conversation).filter_by(
            id=request.conversation_id, session_id=request.session_id
        ).first()
    if not convo:
        convo = Conversation(session_id=request.session_id, title=request.question[:60])
        db.add(convo)
        db.commit()

    result = pipeline.answer(request.question, k=request.k, source_filenames=request.source_filenames)

    db.add(Message(conversation_id=convo.id, role="user", content=request.question,
                   doc_ids_json=json.dumps(request.source_filenames)))
    db.add(Message(conversation_id=convo.id, role="assistant", content=result["answer"],
                   sources_json=json.dumps(result["sources"])))
    db.commit()
    return {**result, "conversation_id": convo.id}

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

def process_uploads_sequentially(saved: list[tuple[str, str]]):
    for doc_id, file_path in saved:
        db = SessionLocal()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        try:
            results, failures = ingest_files([file_path], doc_ids={file_path: doc_id})
            if file_path in results:
                doc.status = "ready"
                doc.chunk_count = results[file_path]
                # ingest_files rewrote bm25_corpus.pkl -- refresh the cached index
                pipeline.reload_bm25()
            else:
                doc.status = "failed"
                doc.error_message = failures.get(file_path, "Ingestion failed -- see server logs.")
        except Exception as e:
            doc.status = "failed"
            doc.error_message = str(e)
        finally:
            db.commit()
            db.close()



@app.get("/documents/{doc_id}")
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc.to_dict()


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.uploaded_at.desc()).all()
    return [d.to_dict() for d in docs]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # The stored file is uploads/{doc_id}{ext}; its basename is the "source"
    # value used in both the Qdrant payload and the BM25 corpus.
    source = os.path.basename(doc.file_path)

    # 1. Remove this document's vectors from Qdrant. Match on doc_id (set by
    #    uploads) OR source (the UUID filename), so legacy points without a
    #    doc_id are still purged.
    try:
        pipeline.qdrant.delete(
            collection_name=config.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        ),
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source),
                        ),
                    ]
                )
            ),
        )
    except Exception as e:
        print(f"Failed to delete Qdrant points for {doc_id}: {e}")

    # 2. Drop this document's chunks from the BM25 corpus and refresh the
    #    in-memory index so keyword search stops returning them too.
    bm25_path = "bm25_corpus.pkl"
    if os.path.exists(bm25_path):
        try:
            with open(bm25_path, "rb") as f:
                corpus = pickle.load(f)
            remaining = [
                c for c in corpus
                if c.get("doc_id") != doc_id and c.get("source") != source
            ]
            with open(bm25_path, "wb") as f:
                pickle.dump(remaining, f)
            pipeline.reload_bm25()
        except Exception as e:
            print(f"Failed to update BM25 corpus for {doc_id}: {e}")

    # 3. Delete the DB row and the stored file
    db.delete(doc)
    db.commit()

    for entry in os.listdir(UPLOAD_DIR):
        if entry.startswith(doc_id):
            try:
                os.remove(os.path.join(UPLOAD_DIR, entry))
            except OSError:
                pass

    return {"id": doc_id, "status": "deleted"}

@app.get("/conversations")
def list_conversations(session_id: str, db: Session = Depends(get_db)):
    rows = (db.query(Conversation).filter_by(session_id=session_id)
            .order_by(Conversation.created_at.desc()).all())
    return [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()} for c in rows]


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, session_id: str, db: Session = Depends(get_db)):
    convo = db.query(Conversation).filter_by(id=conversation_id, session_id=session_id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    msgs = db.query(Message).filter_by(conversation_id=convo.id).order_by(Message.id).all()
    return {
        "id": convo.id,
        "title": convo.title,
        "messages": [{
            "role": m.role,
            "content": m.content,
            "sources": json.loads(m.sources_json) if m.sources_json else None,
            "created_at": m.created_at.isoformat(),
        } for m in msgs],
    }