#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Testing imports...")
print(f"Project root: {PROJECT_ROOT}")
print(f"Python path: {sys.path[:3]}...")

try:
    print("\n1. Testing config import...")
    from backend.config import (
        SUMMARIZER_MODEL_PATH,
        QA_EXTRACTOR_MODEL_PATH,
        TEMP_DIR,
        GEMINI_API_KEY
    )
    print(f"   ✓ Config imported")
    print(f"   Summarizer path: {SUMMARIZER_MODEL_PATH}")
    print(f"   QA path: {QA_EXTRACTOR_MODEL_PATH}")
    print(f"   Summarizer exists: {SUMMARIZER_MODEL_PATH.exists()}")
    print(f"   QA exists: {QA_EXTRACTOR_MODEL_PATH.exists()}")
except Exception as e:
    print(f"   ❌ Config import failed: {e}")
    sys.exit(1)

try:
    print("\n2. Testing service imports...")
    from backend.services.transcription_service import process_source
    from backend.services.summarization_service import generate_hierarchical_summaries
    from backend.services.qa_extraction_service import extract_qa_pairs
    from backend.services.fact_check_service import fact_check_qa_pairs
    print("   ✓ All services imported")
except Exception as e:
    print(f"   ❌ Service import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. Testing FastAPI app import...")
    from backend.app import app
    print("   ✓ FastAPI app imported")
except Exception as e:
    print(f"   ❌ App import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All imports successful!")

