"""Central settings so paths and model names live in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"        # SQL questions (intro databases)
SAMPLE_AI_DIR = DATA_DIR / "sample_ai"  # AI-literacy questions (main demo)
LESSONS_PATH = DATA_DIR / "lessons.json"  # question bank for the 12-lesson curriculum
EEDI_DIR = DATA_DIR / "eedi"
OUTPUT_DIR = ROOT / "outputs"

# Small, fast sentence-embedding model (runs on CPU).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM used to name each misconception and write the fix.
LLM_MODEL = "gemini-2.5-flash"
