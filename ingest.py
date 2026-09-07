import os
import glob
import hashlib
import uuid
import pickle

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

import config


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
            all_chunks.append({"text": chunk.page_content, "source": source, "chunk_index": i})
    return all_chunks


def get_deapi_client() -> OpenAI:
    return OpenAI(base_url=config.DEAPI_BASE_URL, api_key=config.DEAPI_API_KEY)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def point_id(source: str, idx: int, text: str) -> str:
    h = hashlib.md5(f"{source}::{idx}::{text}".encode()).hexdigest()
    return str(uuid.UUID(h))


def main():
    print("Loading + chunking documents...")
    chunks = load_and_chunk_documents(config.DATA_DIR)
    if not chunks:
        print(f"No files found in {config.DATA_DIR}/. Add some and re-run.")
        return
    print(f"Got {len(chunks)} chunks.")

    deapi = get_deapi_client()
    qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    qdrant.recreate_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
    )

    print("Embedding + uploading to Qdrant...")
    batch_size = 16
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(deapi, texts)

        points = [
            PointStruct(
                id=point_id(c["source"], c["chunk_index"], c["text"]),
                vector=emb,
                payload={"text": c["text"], "source": c["source"], "chunk_index": c["chunk_index"]},
            )
            for c, emb in zip(batch, embeddings)
        ]
        qdrant.upsert(collection_name=config.QDRANT_COLLECTION, points=points)

    # Save a local copy of all chunks for BM25 keyword search
    with open("bm25_corpus.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved BM25 keyword index corpus ({len(chunks)} chunks) to bm25_corpus.pkl")

    count = qdrant.count(config.QDRANT_COLLECTION).count
    print(f"Done. {count} chunks stored in Qdrant collection '{config.QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    main()