"""Configuration management for HealthPodcasIQ backend."""
import os
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Load .env from project root
    PROJECT_ROOT = Path(__file__).parent.parent
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# Project root directory (parent of backend/)
PROJECT_ROOT = Path(__file__).parent.parent

# Model paths
SUMMARIZER_MODEL_PATH = PROJECT_ROOT / "models" / "podcastiq-summarizer"
QA_EXTRACTOR_MODEL_PATH = PROJECT_ROOT / "models" / "podcastiq-qa-extractor"

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
BYTEZ_API_KEY = os.getenv("BYTEZ_API_KEY", "")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY", "")
MODEL_PATH_OVERRIDE = os.getenv("MODEL_PATH", None)

# Override model paths if specified
if MODEL_PATH_OVERRIDE:
    SUMMARIZER_MODEL_PATH = Path(MODEL_PATH_OVERRIDE) / "podcastiq-summarizer"
    QA_EXTRACTOR_MODEL_PATH = Path(MODEL_PATH_OVERRIDE) / "podcastiq-qa-extractor"

# API Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Processing Configuration
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "10800"))  # 3 hours default
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # base, small, medium, large
QA_PAIRS_PER_HOUR = int(os.getenv("QA_PAIRS_PER_HOUR", "10"))

# Temporary file storage
TEMP_DIR = PROJECT_ROOT / "backend" / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Validate critical paths
def validate_config():
    """Validate configuration and return any issues."""
    issues = []
    
    if not SUMMARIZER_MODEL_PATH.exists():
        issues.append(f"Summarizer model path does not exist: {SUMMARIZER_MODEL_PATH}")
    
    if not QA_EXTRACTOR_MODEL_PATH.exists():
        issues.append(f"QA extractor model path does not exist: {QA_EXTRACTOR_MODEL_PATH}")
    
    if not TEMP_DIR.exists():
        try:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Could not create temp directory {TEMP_DIR}: {e}")
    
    return issues

