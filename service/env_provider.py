import os

import dotenv

dotenv.load_dotenv()

EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
ENRICHMENT_MODEL = os.getenv("OLLAMA_ENRICHMENT_MODEL", "qwen3:8b")
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", 100))

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "default_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "rag_project")
