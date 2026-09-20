import pickle
import re
import requests
import time

from openai import OpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

import config


# ---------- Clients ----------

def get_deapi_client() -> OpenAI:
    return OpenAI(base_url=config.DEAPI_BASE_URL, api_key=config.DEAPI_API_KEY)


def get_nvidia_client() -> OpenAI:
    return OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY)


# ---------- Vector search ----------

def embed_query(client: OpenAI, query: str) -> list[float]:
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


# ---------- Keyword (BM25) search ----------

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_bm25_index():
    with open("bm25_corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    tokenized = [tokenize(c["text"]) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25, corpus


def bm25_search(bm25: BM25Okapi, corpus: list[dict], query: str, k: int) -> list[dict]:
    scores = bm25.get_scores(tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**corpus[i], "score": scores[i]} for i in ranked_idx]


# ---------- Hybrid fusion ----------

def reciprocal_rank_fusion(vector_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    """
    Merge two ranked lists into one, using RANK POSITION (not raw score,
    since vector similarity and BM25 scores live on different scales).
    """
    scores = {}
    chunk_lookup = {}

    for rank, chunk in enumerate(vector_results):
        key = (chunk["source"], chunk["chunk_index"])
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        chunk_lookup[key] = chunk

    for rank, chunk in enumerate(bm25_results):
        key = (chunk["source"], chunk["chunk_index"])
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        chunk_lookup[key] = chunk

    ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [{**chunk_lookup[key], "fusion_score": round(scores[key], 5)} for key in ranked_keys]


def hybrid_retrieve(qdrant: QdrantClient, deapi: OpenAI, bm25: BM25Okapi, corpus: list[dict],
                     query: str, fetch_k: int = 10, final_k: int = config.TOP_K) -> list[dict]:
    # 1. Vector search — casts a wide net (fetch_k, not final_k)
    query_vector = embed_query(deapi, query)
    vector_hits = qdrant.query_points(
        collection_name=config.QDRANT_COLLECTION, query=query_vector, limit=fetch_k
    ).points
    vector_results = [
        {"text": r.payload["text"], "source": r.payload["source"], "chunk_index": r.payload["chunk_index"]}
        for r in vector_hits
    ]

    # 2. Keyword search — also casts a wide net
    bm25_results = bm25_search(bm25, corpus, query, k=fetch_k)

    # 3. Merge both ranked lists into one
    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    # 4. Trim down to the final number we actually send to the LLM
    return fused[:final_k]


# ---------- Hybrid fusion ----------
def rerank(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    """
    Cross-encoder reranking: instead of scoring the query and each document
    SEPARATELY (like embeddings do, then comparing vectors), a cross-encoder
    looks at the query and one document TOGETHER in a single pass. That joint
    attention is what makes it more precise than embedding similarity — and
    also why it's slower, so we only run it on the small hybrid shortlist,
    never the whole database.
    """
    documents = [c["text"] for c in chunks]

    response = requests.post(
        config.JINA_RERANK_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.JINA_API_KEY}",
        },
        json={
            "model": config.RERANK_MODEL,
            "query": query,
            "top_n": top_n,
            "documents": documents,
            "return_documents": False,
        },
    )
    response.raise_for_status()  # raises an error if the request failed
    results = response.json()["results"]

    reranked = []
    for r in results:
        chunk = chunks[r["index"]]  # Jina returns the ORIGINAL position, not the text itself
        reranked.append({**chunk, "rerank_score": r["relevance_score"]})
    return reranked


# ---------- Prompt + generation ----------

def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = [f"[{i+1}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks)]
    context = "\n\n".join(context_blocks)

    return f"""You are answering a question using ONLY the context provided below.

Context:
{context}

Question: {query}

Instructions:
- Answer using only information in the context above.
- If the context doesn't contain enough information to answer, say so explicitly — do not guess.
- Cite sources inline using bracket numbers, e.g. [1], [2], matching the context blocks above.
"""


def generate_answer(nvidia: OpenAI, prompt: str) -> str:
    response = nvidia.chat.completions.create(
        model=config.GENERATION_MODEL,
        max_tokens=config.MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ---------- Orchestration ----------

def answer_question(query: str, k: int = config.TOP_K) -> dict:
    t0 = time.time()
    deapi = get_deapi_client()
    nvidia = get_nvidia_client()
    qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    bm25, corpus = load_bm25_index()
    print(f"[setup]     {time.time() - t0:.2f}s")

    t1 = time.time()
    candidates = hybrid_retrieve(qdrant, deapi, bm25, corpus, query, fetch_k=10, final_k=10)
    print(f"[retrieval] {time.time() - t1:.2f}s")

    t2 = time.time()
    chunks = rerank(query, candidates, top_n=k)
    print(f"[rerank]    {time.time() - t2:.2f}s")

    if not chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    t3 = time.time()
    prompt = build_prompt(query, chunks)
    answer = generate_answer(nvidia, prompt)
    print(f"[generate]  {time.time() - t3:.2f}s")

    print(f"[TOTAL]     {time.time() - t0:.2f}s")

    return {
        "answer": answer,
        "sources": [
            {"n": i + 1, "source": c["source"], "rerank_score": round(c["rerank_score"], 4)}
            for i, c in enumerate(chunks)
        ],
    }


if __name__ == "__main__":
    print("RAG assistant ready. Ask a question (or 'quit').\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        result = answer_question(q)
        print(f"\nAssistant: {result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['source']}")
        print()