import os
import glob
import hashlib
import uuid
import pickle
import re
import time

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI, RateLimitError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

import config
from extract_tables import extract_all_tables


def is_noise(text: str) -> bool:
    """Drop chunks that hurt retrieval: figure debris and reference entries."""
    if text.count("<pad>") > 2:
        return True
    if len(re.findall(r"^\[\d+\]", text, flags=re.MULTILINE)) >= 2:
        return True
    return False


def is_probably_table(text: str) -> bool:
    """
    Heuristic: a chunk that's mostly numbers (a garbled, column-less table
    extraction) rather than prose. These get replaced by vision-transcribed
    markdown tables instead, so the raw broken version is dropped.
    """
    tokens = text.split()
    if not tokens:
        return False
    numeric_tokens = sum(1 for t in tokens if re.match(r"^[\d.,·%]+$", t))
    return (numeric_tokens / len(tokens)) > 0.3


def load_and_chunk_documents(data_dir: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    all_chunks = []
    for path in glob.glob(os.path.join(data_dir, "*")):
        ext = os.path.splitext(path)[1].lower()
        source = os.path.basename(path)
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext in (".txt", ".md"):
            loader = TextLoader(path, encoding="utf-8")
        else:
            continue
        chunks = splitter.split_documents(loader.load())
        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            if is_noise(text) or is_probably_table(text):
                continue
            all_chunks.append({"text": text, "source": source, "chunk_index": i, "type": "text"})
    return all_chunks


def load_tables_for_pdfs(data_dir: str) -> list[dict]:
    """Run vision-based table transcription on every PDF in data_dir."""
    table_chunks = []
    for path in glob.glob(os.path.join(data_dir, "*.pdf")):
        source = os.path.basename(path)
        print(f"Extracting tables from {source}...")
        tables = extract_all_tables(path)
        for i, t in enumerate(tables):
            table_chunks.append({
                "text": t["text"],
                "source": source,
                "chunk_index": f"table_{i}",
                "type": "table",
                "page": t["page"],
            })
    return table_chunks


def get_deapi_client() -> OpenAI:
    return OpenAI(base_url=config.DEAPI_BASE_URL, api_key=config.DEAPI_API_KEY)


def embed_texts(client: OpenAI, texts: list[str], max_retries: int = 5) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
            return [item.embedding for item in response.data]
        except RateLimitError as e:
            wait_time = 20 * (attempt + 1)
            print(f"  Rate limited: {e}")
            print(f"  Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)
    raise RuntimeError("Exceeded max retries for embedding request.")


def point_id(source: str, idx, text: str) -> str:
    h = hashlib.md5(f"{source}::{idx}::{text}".encode()).hexdigest()
    return str(uuid.UUID(h))


def main():
    print("Loading + chunking documents...")
    text_chunks = load_and_chunk_documents(config.DATA_DIR)
    print(f"Got {len(text_chunks)} text chunks (after filtering noise + garbled tables).")

    table_chunks = load_tables_for_pdfs(config.DATA_DIR)
    print(f"Got {len(table_chunks)} vision-transcribed table chunks.")

    chunks = text_chunks + table_chunks
    if not chunks:
        print(f"No content found in {config.DATA_DIR}/. Add some and re-run.")
        return

    deapi = get_deapi_client()
    qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    qdrant.recreate_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
    )

    print("Embedding + uploading to Qdrant...")
    batch_size = config.EMBED_BATCH_SIZE
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(deapi, texts)

        points = [
            PointStruct(
                id=point_id(c["source"], c["chunk_index"], c["text"]),
                vector=emb,
                payload={
                    "text": c["text"],
                    "source": c["source"],
                    "chunk_index": c["chunk_index"],
                    "type": c["type"],
                    "page": c.get("page"),
                },
            )
            for c, emb in zip(batch, embeddings)
        ]
        qdrant.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
        time.sleep(config.EMBED_BATCH_DELAY)

    with open("bm25_corpus.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved BM25 keyword index corpus ({len(chunks)} chunks) to bm25_corpus.pkl")

    count = qdrant.count(config.QDRANT_COLLECTION).count
    print(f"Done. {count} chunks stored in Qdrant collection '{config.QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    main()