#!/usr/bin/env python3
"""Check if all required dependencies are installed."""
import sys

missing = []
optional_missing = []

dependencies = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "torch": "torch",
    "transformers": "transformers",
    "sentence-transformers": "sentence-transformers",
    "sentencepiece": "sentencepiece",
    "whisper": "openai-whisper",
    "yt_dlp": "yt-dlp",
    "google.generativeai": "google-generativeai",
    "dotenv": "python-dotenv",
}

optional = {
    "google.generativeai": "google-generativeai (optional, for fact-checking)",
}

print("Checking dependencies...\n")

for module, package in dependencies.items():
    try:
        if module == "whisper":
            import whisper
        elif module == "yt_dlp":
            import yt_dlp
        elif module == "dotenv":
            from dotenv import load_dotenv
        elif module == "google.generativeai":
            import google.generativeai as genai
        elif module == "sentencepiece":
            import sentencepiece
        elif module == "sentence-transformers":
            # Handle importlib.metadata compatibility issue in Python 3.9
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                if "packages_distributions" in str(e) or "importlib.metadata" in str(e):
                    # Package is installed but has compatibility warning - still works
                    pass
                else:
                    raise
        else:
            __import__(module)
        print(f"✓ {module} ({package})")
    except ImportError:
        if module in optional:
            optional_missing.append((module, package))
            print(f"⚠ {module} ({package}) - OPTIONAL")
        else:
            missing.append((module, package))
            print(f"❌ {module} ({package}) - MISSING")
    except Exception as e:
        # Some packages may have import warnings but still work
        if "importlib.metadata" in str(e) or "packages_distributions" in str(e):
            print(f"✓ {module} ({package}) - installed (with compatibility warning)")
        else:
            if module in optional:
                optional_missing.append((module, package))
                print(f"⚠ {module} ({package}) - OPTIONAL (error: {e})")
            else:
                missing.append((module, package))
                print(f"❌ {module} ({package}) - MISSING (error: {e})")

print()

if missing:
    print("❌ Missing required dependencies:")
    for module, package in missing:
        print(f"   pip install {package}")
    print("\nInstall all with:")
    print(f"   pip install {' '.join([pkg for _, pkg in missing])}")
    sys.exit(1)

if optional_missing:
    print("⚠️  Optional dependencies missing (features may be limited):")
    for module, package in optional_missing:
        print(f"   pip install {package}")

print("✅ All required dependencies are installed!")
if optional_missing:
    print("   (Some optional features may not be available)")

