import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "ollama")

OLLAMA_MODEL: str = "llama3.2"
OPENAI_MODEL: str = "gpt-4o-mini"
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

WHISPER_MODEL_SIZE: str = "base"

try:
    import torch
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE: str = "cpu"

BASE_DIR: Path = Path(__file__).parent
TRANSCRIPTS_DIR: Path = BASE_DIR / "transcripts"
INDEXES_DIR: Path = BASE_DIR / "indexes"

TRANSCRIPTS_DIR.mkdir(exist_ok=True)
INDEXES_DIR.mkdir(exist_ok=True)
