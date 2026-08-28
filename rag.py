from openai import OpenAI
from qdrant_client import QdrantClient

import config


def get_deapi_client() -> OpenAI:
    return OpenAI(base_url=config.DEAPI_BASE_URL, api_key=config.DEAPI_API_KEY)


def get_nvidia_client() -> OpenAI:
    return OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY)


def embed_query(client: OpenAI, query: str) -> list[float]:
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def retrieve(qdrant: QdrantClient, deapi: OpenAI, query: str, k: int = config.TOP_K) -> list[dict]:
    query_vector = embed_query(deapi, query)
    results = qdrant.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=k,
    ).points

    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "chunk_index": r.payload["chunk_index"],
            "score": r.score,
        }
        for r in results
    ]


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


def answer_question(query: str, k: int = config.TOP_K) -> dict:
    deapi = get_deapi_client()
    nvidia = get_nvidia_client()
    qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    chunks = retrieve(qdrant, deapi, query, k=k)
    if not chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    prompt = build_prompt(query, chunks)
    answer = generate_answer(nvidia, prompt)

    return {
        "answer": answer,
        "sources": [{"n": i+1, "source": c["source"], "score": round(c["score"], 4)} for i, c in enumerate(chunks)],
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
            print(f"  [{s['n']}] {s['source']} (score={s['score']})")
        print()