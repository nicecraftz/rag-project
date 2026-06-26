import dotenv
import os

dotenv.load_dotenv()

EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
ENRICHMENT_MODEL = os.getenv("OLLAMA_ENRICHMENT_MODEL", "qwen3:8b")
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", 100))