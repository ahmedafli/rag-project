from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import RAGPipeline

app = FastAPI(title="RAG Assistant API")

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