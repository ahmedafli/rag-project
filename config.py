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
GENERATION_MODEL = "minimaxai/minimax-m3"
MAX_TOKENS = 1024

# --- Vector DB: Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "documents"

# --- Data ---
DATA_DIR = "data"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval ---
TOP_K = 5