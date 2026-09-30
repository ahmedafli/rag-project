import os
from dotenv import load_dotenv

load_dotenv()

# --- Embedding provider: deAPI (OpenAI-compatible gateway) ---
DEAPI_API_KEY = os.getenv("DEAPI_API_KEY")
DEAPI_BASE_URL = "https://oai.deapi.ai/v1"
EMBEDDING_MODEL = "Bge_M3_FP16"
EMBEDDING_DIM = 1024

# --- LLM provider: NVIDIA NIM ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GENERATION_MODEL = "nvidia/nemotron-3-super-120b-a12b"
MAX_TOKENS = 1024

# --- Vector DB: Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "documents"

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
RERANK_MODEL = "jina-reranker-v3.5"

# --- Data ---
DATA_DIR = "data"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval ---
TOP_K = 5



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "qwen/qwen3.8-27b"

EMBED_BATCH_SIZE = 8
EMBED_BATCH_DELAY = 15  # seconds to wait between each embedding batch

VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VISION_MODEL = "qwen/qwen3.8-27b:free"