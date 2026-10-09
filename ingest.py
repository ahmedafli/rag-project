import os
import glob
import hashlib
import logging
import uuid
import pickle
import re
import time

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from tqdm import tqdm

import config
from extract_tables import extract_all_tables


logger = logging.getLogger("ingest")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def ensure_source_index(qdrant: QdrantClient) -> None:
    """Create a keyword payload index on 'source' so it can be filtered on."""
    qdrant.create_payload_index(
        collection_name=config.QDRANT_COLLECTION,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD,
    )


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
            all_chunks.append({
                "text": text, "source": source, "doc_id": os.path.splitext(source)[0],
                "chunk_index": i, "type": "text",
            })
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
                "doc_id": os.path.splitext(source)[0],
                "chunk_index": f"table_{i}",
                "type": "table",
                "page": t["page"],
            })
    return table_chunks


def get_deapi_client() -> OpenAI:
    return OpenAI(
        base_url=config.DEAPI_BASE_URL,
        api_key=config.DEAPI_API_KEY,
        timeout=config.EMBED_TIMEOUT,
    )


# Network-ish failures worth retrying: SDK exceptions plus the raw socket/urllib3
# timeouts that can leak through as plain TimeoutError/OSError.
TRANSIENT_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
    TimeoutError,
    ConnectionError,
    OSError,
)


def embed_texts(client: OpenAI, texts: list[str], max_retries: int = 5) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
            return [item.embedding for item in response.data]
        except TRANSIENT_ERRORS as e:
            wait_time = 20 * (attempt + 1)
            logger.warning(
                "embed_texts: %s: %s -- retry %d/%d in %ds",
                type(e).__name__, e, attempt + 1, max_retries, wait_time,
            )
            time.sleep(wait_time)
    raise RuntimeError("Exceeded max retries for embedding request.")


def point_id(source: str, idx, text: str) -> str:
    h = hashlib.md5(f"{source}::{idx}::{text}".encode()).hexdigest()
    return str(uuid.UUID(h))

def ingest_files(file_paths: list[str], doc_ids: dict[str, str] | None = None) -> tuple[dict, dict]:
    """
    Ingest one or more files, APPENDING to whatever's already in Qdrant and
    the BM25 corpus, instead of recreating/overwriting them. Used by the
    upload endpoint -- unlike main(), this never wipes existing data.

    doc_ids optionally maps a file path to its document id (e.g. the DB row
    id). When absent, the source filename stem is used.

    Returns (results, failures):
      results  = {file_path: chunk_count} for files that succeeded
      failures = {file_path: "stage: error message"} for files that failed
    """
    doc_ids = doc_ids or {}
    deapi = get_deapi_client()
    qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    # Create the collection only if it doesn't exist yet -- never recreate here
    existing = [c.name for c in qdrant.get_collections().collections]
    if config.QDRANT_COLLECTION not in existing:
        logger.info("Creating Qdrant collection '%s'", config.QDRANT_COLLECTION)
        qdrant.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        )
        ensure_source_index(qdrant)

    # Load the existing BM25 corpus, if any, so we can append to it
    try:
        with open("bm25_corpus.pkl", "rb") as f:
            full_corpus = pickle.load(f)
    except FileNotFoundError:
        full_corpus = []

    results = {}
    failures = {}

    for path in file_paths:
        source = os.path.basename(path)
        doc_id = doc_ids.get(path) or os.path.splitext(source)[0]
        ext = os.path.splitext(path)[1].lower()
        t_start = time.time()
        stage = "start"

        try:
            # --- text chunks ---
            stage = "load + chunk"
            logger.info("[%s] loading and chunking (%s)...", source, ext or "no ext")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            )
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext in (".txt", ".md"):
                loader = TextLoader(path, encoding="utf-8")
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            raw_chunks = splitter.split_documents(loader.load())
            file_chunks = []
            for i, chunk in enumerate(raw_chunks):
                text = chunk.page_content
                if is_noise(text) or is_probably_table(text):
                    continue
                file_chunks.append({
                    "text": text, "source": source, "doc_id": doc_id,
                    "chunk_index": i, "type": "text",
                })
            logger.info(
                "[%s] chunked: %d kept / %d raw text chunks",
                source, len(file_chunks), len(raw_chunks),
            )

            # --- table chunks (vision transcription), PDFs only ---
            if ext == ".pdf":
                stage = "table extraction"
                logger.info("[%s] scanning for tables (vision)...", source)
                tables = extract_all_tables(path)
                for i, t in enumerate(tables):
                    file_chunks.append({
                        "text": t["text"], "source": source, "doc_id": doc_id,
                        "chunk_index": f"table_{i}", "type": "table", "page": t["page"],
                    })
                logger.info("[%s] table chunks added: %d", source, len(tables))

            if not file_chunks:
                raise ValueError("No extractable content found in file.")

            # --- embed + upload this file's chunks only ---
            stage = "embedding + upload"
            batch_size = config.EMBED_BATCH_SIZE
            total_batches = (len(file_chunks) + batch_size - 1) // batch_size
            logger.info(
                "[%s] embedding %d chunks in %d batch(es) of %d (delay %ss between batches)",
                source, len(file_chunks), total_batches, batch_size, config.EMBED_BATCH_DELAY,
            )
            for i in range(0, len(file_chunks), batch_size):
                batch = file_chunks[i:i + batch_size]
                batch_no = i // batch_size + 1
                texts = [c["text"] for c in batch]
                logger.info("[%s] batch %d/%d: requesting %d embeddings...", source, batch_no, total_batches, len(texts))
                t_embed = time.time()
                embeddings = embed_texts(deapi, texts)
                logger.info("[%s] batch %d/%d: embedded in %.2fs, upserting...", source, batch_no, total_batches, time.time() - t_embed)
                points = [
                    PointStruct(
                        id=point_id(c["source"], c["chunk_index"], c["text"]),
                        vector=emb,
                        payload={
                            "text": c["text"], "source": c["source"], "doc_id": c["doc_id"],
                            "chunk_index": c["chunk_index"],
                            "type": c["type"], "page": c.get("page"),
                        },
                    )
                    for c, emb in zip(batch, embeddings)
                ]
                qdrant.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
                logger.info("[%s] batch %d/%d: upserted (%d/%d chunks done)", source, batch_no, total_batches, min(i + batch_size, len(file_chunks)), len(file_chunks))
                if i + batch_size < len(file_chunks):
                    time.sleep(config.EMBED_BATCH_DELAY)

            full_corpus.extend(file_chunks)
            results[path] = len(file_chunks)
            logger.info("[%s] DONE: %d chunks in %.2fs", source, len(file_chunks), time.time() - t_start)

        except Exception as e:
            logger.exception("[%s] FAILED during '%s' after %.2fs: %s", source, stage, time.time() - t_start, e)
            failures[path] = f"{stage}: {type(e).__name__}: {e}"

    # Re-save the full corpus once, after all files processed
    with open("bm25_corpus.pkl", "wb") as f:
        pickle.dump(full_corpus, f)

    return results, failures


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
    ensure_source_index(qdrant)

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
                    "doc_id": c["doc_id"],
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