"""FastAPI application for HealthPodcasIQ backend."""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import re
import sys
import shutil
from pathlib import Path

# Ensure backend package is in Python path BEFORE importing backend modules
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import backend modules
try:
    from backend.config import (
        CORS_ORIGINS, 
        SUMMARIZER_MODEL_PATH, 
        QA_EXTRACTOR_MODEL_PATH,
        GEMINI_API_KEY,
        BYTEZ_API_KEY,
        SAMBANOVA_API_KEY,
        TEMP_DIR,
        WHISPER_MODEL
    )
    from backend.services.transcription_service import process_source
    from backend.services.summarization_service import generate_hierarchical_summaries
    from backend.services.qa_extraction_service import extract_qa_pairs
    from backend.services.fact_check_service import fact_check_qa_pairs
    from backend.database import get_cached_result, save_processing_result, get_history
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import backend modules: {e}")
    print(f"   Current working directory: {Path.cwd()}")
    print(f"   Python path: {sys.path}")
    print(f"   Backend dir: {BACKEND_DIR}")
    print(f"   Project root: {PROJECT_ROOT}")
    raise


def validate_startup():
    """Validate configuration and models on startup."""
    errors = []
    warnings = []
    
    print("\n" + "="*60)
    print("HealthPodcasIQ Backend - Startup Validation")
    print("="*60)
    
    # Check model paths
    print("\n[1/4] Checking model paths...")
    if not SUMMARIZER_MODEL_PATH.exists():
        errors.append(f"Summarizer model not found at: {SUMMARIZER_MODEL_PATH}")
        print(f"  ❌ Summarizer model missing: {SUMMARIZER_MODEL_PATH}")
    else:
        print(f"  ✓ Summarizer model found: {SUMMARIZER_MODEL_PATH}")
    
    if not QA_EXTRACTOR_MODEL_PATH.exists():
        errors.append(f"QA extractor model not found at: {QA_EXTRACTOR_MODEL_PATH}")
        print(f"  ❌ QA extractor model missing: {QA_EXTRACTOR_MODEL_PATH}")
    else:
        print(f"  ✓ QA extractor model found: {QA_EXTRACTOR_MODEL_PATH}")
    
    # Check required model files
    print("\n[2/4] Checking required model files...")
    required_summarizer_files = ["config.json", "model.safetensors"]
    # Check for tokenizer.json OR tokenizer_config.json
    if (SUMMARIZER_MODEL_PATH / "tokenizer.json").exists():
        required_summarizer_files.append("tokenizer.json")
    elif (SUMMARIZER_MODEL_PATH / "tokenizer_config.json").exists():
        required_summarizer_files.append("tokenizer_config.json")
    
    for file in required_summarizer_files:
        file_path = SUMMARIZER_MODEL_PATH / file
        if not file_path.exists():
            errors.append(f"Missing summarizer file: {file}")
            print(f"  ❌ Missing: {file}")
        else:
            print(f"  ✓ Found: {file}")
    
    required_qa_files = ["config.json", "model.safetensors", "tokenizer_config.json"]
    for file in required_qa_files:
        file_path = QA_EXTRACTOR_MODEL_PATH / file
        if not file_path.exists():
            errors.append(f"Missing QA extractor file: {file}")
            print(f"  ❌ Missing: {file}")
        else:
            print(f"  ✓ Found: {file}")
    
    # Check environment variables
    print("\n[3/4] Checking environment variables...")
    if not GEMINI_API_KEY:
        warnings.append("GEMINI_API_KEY not set - external fact-checking will be disabled")
        print("  ⚠️  GEMINI_API_KEY not set (external fact-checking disabled)")
    else:
        print("  ✓ GEMINI_API_KEY is set")
    
    # Check directories
    print("\n[4/4] Checking directories...")
    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created temp directory: {TEMP_DIR}")
    else:
        print(f"  ✓ Temp directory exists: {TEMP_DIR}")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print("❌ STARTUP VALIDATION FAILED")
        print("="*60)
        for error in errors:
            print(f"  ERROR: {error}")
        print("\n⚠️  Backend will start but may fail during processing.")
        print("   Please fix the errors above before processing podcasts.")
    elif warnings:
        print("⚠️  STARTUP VALIDATION COMPLETE (with warnings)")
        print("="*60)
        for warning in warnings:
            print(f"  WARNING: {warning}")
    else:
        print("✓ STARTUP VALIDATION SUCCESSFUL")
        print("="*60)
    print()
    
    return len(errors) == 0


# Use lifespan for FastAPI (works with both old and new versions)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    print("🚀 Starting HealthPodcasIQ Backend...")
    validate_startup()
    yield
    # Shutdown (if needed)
    print("🛑 Shutting down HealthPodcasIQ Backend...")


app = FastAPI(
    title="HealthPodcasIQ API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ProcessPodcastRequest(BaseModel):
    source: str  # YouTube URL or file path

class CreateSequenceRequest(BaseModel):
    sources: List[str]  # Array of YouTube URLs


class ProcessPodcastFileRequest(BaseModel):
    """Request model for file upload processing."""
    pass


class HealthResponse(BaseModel):
    status: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    error_type: Optional[str] = None
    suggestion: Optional[str] = None


def extract_title_from_transcript(transcript: str) -> str:
    """
    Extract a title from transcript (first sentence or first 100 chars).
    
    Args:
        transcript: Full transcript text
        
    Returns:
        Title string
    """
    # Try to get first sentence
    sentences = re.split(r'[.!?]+', transcript)
    if sentences and len(sentences[0].strip()) > 10:
        title = sentences[0].strip()
        if len(title) > 100:
            title = title[:100] + "..."
        return title
    
    # Fallback to first 100 characters
    return transcript[:100].strip() + "..."



@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Quick validation check
    models_ok = SUMMARIZER_MODEL_PATH.exists() and QA_EXTRACTOR_MODEL_PATH.exists()
    status = "healthy" if models_ok else "degraded"
    message = "HealthPodcasIQ backend is running" if models_ok else "Backend running but models may be missing"
    
    return HealthResponse(status=status, message=message)


@app.post("/process-podcast-file")
async def process_podcast_file(file: UploadFile = File(...)):
    """
    Process a podcast from uploaded audio/video file.
    
    Accepts multipart/form-data with file upload.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    allowed_extensions = {'.mp3', '.mp4', '.wav', '.m4a', '.ogg', '.flac', '.webm'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    temp_file_path = TEMP_DIR / f"upload_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file
        return await process_podcast(ProcessPodcastRequest(source=str(temp_file_path)))
    finally:
        # Clean up temp file
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_file_path}: {e}")


@app.post("/process-podcast")
async def process_podcast(request: ProcessPodcastRequest):
    """
    Process a health/fitness/nutrition podcast from YouTube URL or file path.
    
    Pipeline:
    1. Transcribe audio using Whisper
    2. Generate hierarchical summaries using trained BART model
    3. Extract health/fitness/nutrition Q&A pairs using trained T5 model
    4. Fact-check claims using two-layer verification (internal transcript + external Gemini API)
    
    Returns:
        SummaryResult with title, summaries, and verified Q&A pairs
    """
    try:
        print(f"Processing health/fitness/nutrition podcast from: {request.source}")
        print(f"{'='*60}\n")
        
        # Step 0: Check Cache
        cached_result = get_cached_result(request.source)
        if cached_result:
            print("✓ Found cached result! Returning immediately.")
            return cached_result
        
        # Step 1: Transcribe audio
        print("Step 1: Transcribing audio with Whisper...")
        transcript, thumbnail_url = process_source(request.source)
        
        if not transcript or len(transcript.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Transcript is too short or empty. Please check the audio source."
            )
        
        print(f"✓ Transcript generated: {len(transcript)} characters, {len(transcript.split())} words")
        
        # Step 2: Generate title
        title = extract_title_from_transcript(transcript)
        print(f"✓ Extracted title: {title}")
        
        # Step 3: Generate hierarchical summaries using trained BART model
        print("\nStep 2: Generating hierarchical summaries using trained BART model...")
        try:
            summaries = generate_hierarchical_summaries(transcript)
            print(f"✓ Summaries generated: short ({len(summaries['short'])} chars), "
                  f"medium ({len(summaries['medium'])} chars), "
                  f"long ({len(summaries['long'])} chars)")
        except FileNotFoundError as e:
            error_msg = str(e)
            print(f"❌ Model not found: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Summarization model not found: {error_msg}. Please ensure the model is installed in {SUMMARIZER_MODEL_PATH}"
            )
        except RuntimeError as e:
            error_msg = str(e)
            print(f"❌ Model loading error: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load summarization model: {error_msg}"
            )
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error generating summaries: {error_msg}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate summaries: {error_msg}"
            )
        
        # Step 4: Extract Q&A pairs using trained T5 model
        print("\nStep 3: Extracting health/fitness/nutrition Q&A pairs using trained T5 model...")
        try:
            # Estimate duration: assume ~150 words per minute
            estimated_duration_hours = len(transcript.split()) / (150 * 60)
            qa_pairs = extract_qa_pairs(transcript, estimated_duration_hours)
            
            if not qa_pairs:
                print("⚠️  Warning: No health/fitness/nutrition Q&A pairs extracted.")
                print("This indicates the content is not health/fitness/nutrition related.")
                print("The service only generates Q&A pairs for health-related content.")
            elif len(qa_pairs) < 10:
                print(f"⚠️  Warning: Only {len(qa_pairs)} health Q&A pairs extracted (target was at least 10).")
                print("This may indicate the content has limited health/fitness/nutrition focus.")
            else:
                print(f"✓ Extracted {len(qa_pairs)} health/fitness/nutrition Q&A pairs")
        except FileNotFoundError as e:
            error_msg = str(e)
            print(f"❌ Model not found: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"QA extraction model not found: {error_msg}. Please ensure the model is installed in {QA_EXTRACTOR_MODEL_PATH}"
            )
        except RuntimeError as e:
            error_msg = str(e)
            print(f"❌ Model loading error: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load QA extraction model: {error_msg}"
            )
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error extracting Q&A pairs: {error_msg}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract Q&A pairs: {error_msg}"
            )
        
        # Step 5: Fact-check Q&A pairs (two-layer: internal transcript + external Gemini)
        print("\nStep 4: Fact-checking claims (internal transcript verification + external Gemini API)...")
        try:
            qa_pairs = fact_check_qa_pairs(qa_pairs, transcript)
            print(f"✓ Fact-checking complete for {len(qa_pairs)} Q&A pairs")
        except Exception as e:
            print(f"Warning: Fact-checking encountered errors: {e}")
            # Continue without fact-checking if it fails
            import traceback
            traceback.print_exc()
        
        # Step 6: Format response
        result = {
            "title": title,
            "thumbnailUrl": thumbnail_url,
            "overallSummary": {
                "short": summaries["short"],
                "medium": summaries["medium"],
                "long": summaries["long"]
            },
            "qaPairs": qa_pairs
        }
        
        # Step 7: Save to Database for Caching
        print("\nStep 5: Saving result to database...")
        save_processing_result(
            request.source, 
            title, 
            thumbnail_url, 
            transcript, 
            {
                "short": summaries["short"],
                "medium": summaries["medium"],
                "long": summaries["long"]
            }, 
            qa_pairs
        )
        
        print(f"\n{'='*60}")
        print("✓ Processing complete!")
        print(f"{'='*60}\n")
        
        return result
        
    except FileNotFoundError as e:
        error_msg = str(e)
        raise HTTPException(
            status_code=404,
            detail=error_msg
        )
    except ValueError as e:
        error_msg = str(e)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {error_msg}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"\n❌ Error processing podcast: {error_type}: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Provide helpful error messages based on error type
        if "model" in error_msg.lower() or "Model" in error_type:
            detail = f"Model loading error: {error_msg}. Please ensure models are properly installed."
            suggestion = "Check that model files exist in the models/ directory."
        elif "whisper" in error_msg.lower() or "transcription" in error_msg.lower():
            detail = f"Transcription error: {error_msg}"
            suggestion = "Check audio file format and ensure Whisper model is available."
        elif "gemini" in error_msg.lower() or "api" in error_msg.lower():
            detail = f"API error: {error_msg}"
            suggestion = "Check your GEMINI_API_KEY in the .env file."
        else:
            detail = f"Processing error: {error_msg}"
            suggestion = "Check backend logs for more details."
        
        raise HTTPException(
            status_code=500,
            detail=f"{detail} {suggestion}" if suggestion else detail
        )


async def _create_sequence_internal(sources: List[str]):
    """
    Internal logic to create an episode sequence from multiple health/fitness/nutrition podcasts.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Creating episode sequence from {len(sources)} podcasts")
        print(f"{'='*60}\n")
        
        if len(sources) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 podcast sources are required for episode sequencing"
            )
        
        # Step 1: Process each podcast (transcription + summarization)
        print("Step 1: Processing individual podcasts...")
        processed_podcasts = []
        
        for i, source in enumerate(sources, 1):
            print(f"\nProcessing Podcast {i}/{len(sources)}: {source}")
            try:
                # Transcribe
                transcript, _ = process_source(source)
                if not transcript or len(transcript.strip()) < 100:
                    print(f"⚠️  Warning: Podcast {i} transcript too short, skipping")
                    continue
                
                # Generate summaries (we'll use medium summary for synthesis)
                summaries = generate_hierarchical_summaries(transcript)
                
                processed_podcasts.append({
                    "id": i,
                    "source": source,
                    "transcript": transcript,
                    "summary": summaries["medium"]
                })
                print(f"✓ Podcast {i} processed: {len(transcript)} chars, {len(transcript.split())} words")
            except Exception as e:
                print(f"⚠️  Warning: Failed to process Podcast {i}: {e}")
                continue
        
        if len(processed_podcasts) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Could not process enough podcasts. Only {len(processed_podcasts)} succeeded."
            )
        
        print(f"\n✓ Successfully processed {len(processed_podcasts)} podcasts")
        
        # Step 2: Synthesize segments using Bytez/SambaNova API
        api_providers = []
        if BYTEZ_API_KEY: api_providers.append("Bytez")
        if SAMBANOVA_API_KEY: api_providers.append("SambaNova")
        
        provider_str = " + ".join(api_providers) if api_providers else "None"
        print(f"\nStep 2: Synthesizing episode sequence using {provider_str} API...")
        
        try:
            import json
            import requests
            
            # Build prompt with summaries (TOKEN OPTIMIZATION)
            summaries_text = "\n\n".join([
                f"**Podcast {podcast['id']} Summary:**\n{podcast['summary']}"
                for podcast in processed_podcasts
            ])
            
            prompt = f"""You are an expert curriculum designer. Synthesize these podcast summaries into a single, coherent 'episode sequence'.

**Instructions:**
1. Sequence segments logically: Foundations FIRST, advanced topics LATER.
2. Each segment must cover a core concept (e.g., 'Calorie Deficit', 'Protein').
3. For each segment, provide: Title, 2-3 sentence summary, Original Podcast #, and Key Concept.
4. Total: 5-8 segments across all podcasts.
5. Generate an overall Title and a 2-3 sentence Introduction.
6. Return ONLY valid JSON in this format:

{{
  "sequenceTitle": "Title",
  "sequenceIntroduction": "Intro",
  "segments": [
    {{
      "id": "seg1",
      "title": "Segment Title",
      "summary": "Summary text",
      "sourcePodcast": 1,
      "keyConcept": "Concept Name"
    }}
  ]
}}

**Podcast Summaries:**
{summaries_text}

Return JSON:"""
            
            response_text = ""
            
            # Try Bytez API first
            if BYTEZ_API_KEY:
                try:
                    url = "https://api.bytez.com/models/v2/openai/v1/chat/completions"
                    headers = {
                        "Authorization": BYTEZ_API_KEY,
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "Qwen/Qwen3-1.7B",
                        "messages": [
                            {"role": "system", "content": "You are an expert curriculum designer for health/fitness content. Always return valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 2000,
                        "temperature": 0.3
                    }
                    
                    resp = requests.post(url, headers=headers, json=payload, timeout=60)
                    resp.raise_for_status()
                    result_data = resp.json()
                    response_text = result_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print("  Using Bytez API")
                except Exception as bytez_error:
                    print(f"  Bytez API error: {bytez_error}, falling back to SambaNova")
                    response_text = ""
            
            # Fallback to SambaNova if Bytez failed or not configured
            if not response_text and SAMBANOVA_API_KEY:
                try:
                    url = "https://api.sambanova.ai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "DeepSeek-R1-Distill-Llama-70B",
                        "messages": [
                            {"role": "system", "content": "You are an expert curriculum designer for health/fitness content. Always return valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 2000,
                        "temperature": 0.1
                    }
                    
                    resp = requests.post(url, headers=headers, json=payload, timeout=60)
                    resp.raise_for_status()
                    result_data = resp.json()
                    response_text = result_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print("  Using SambaNova API")
                except Exception as sambanova_error:
                    print(f"  SambaNova API error: {sambanova_error}")
                    raise ValueError(f"Both Bytez and SambaNova APIs failed. Bytez: {bytez_error if 'bytez_error' in dir() else 'not configured'}, SambaNova: {sambanova_error}")
            
            if not response_text:
                raise ValueError("No API key configured. Please set BYTEZ_API_KEY or SAMBANOVA_API_KEY in your .env file.")

            
            # Clean up response (handle thinking tags and markdown)
            # Find the first '{' and last '}'
            try:
                start_index = response_text.find('{')
                end_index = response_text.rfind('}')
                
                if start_index != -1 and end_index != -1:
                    json_str = response_text[start_index:end_index+1]
                else:
                    json_str = response_text
                
                # Further cleaning for markdown
                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                if json_str.startswith("```"):
                    json_str = json_str[3:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                json_str = json_str.strip()
                
                result = json.loads(json_str)
            except Exception as parse_error:
                print(f"❌ Initial JSON parsing failed: {parse_error}")
                print(f"   Response text snippet: {response_text[:500]}...")
                # Try fallback: just the raw text if no braces found
                result = json.loads(response_text.strip())
            
            # Validate and format result
            if "segments" not in result or not isinstance(result["segments"], list):
                raise ValueError("Invalid response format: missing segments")
            
            # Ensure all segments have required fields
            for segment in result["segments"]:
                if "id" not in segment:
                    segment["id"] = f"seg{result['segments'].index(segment) + 1}"
                if "sourcePodcast" not in segment:
                    segment["sourcePodcast"] = 1
            
            print(f"✓ Generated sequence with {len(result['segments'])} segments")
            print(f"  Title: {result.get('sequenceTitle', 'Untitled')}")
            
            return result
            
        except ValueError as e:
            if "API" in str(e) or "configured" in str(e):
                raise HTTPException(
                    status_code=500,
                    detail=str(e)
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to synthesize sequence: {str(e)}"
            )
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response text: {response_text[:500]}...")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse sequence synthesis response. Please try again."
            )
        except Exception as e:
            raise e # Let the outer block catch it
            
    except HTTPException:
        raise
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"\n❌ Error creating sequence: {error_type}: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Sequence creation error: {error_msg}"
        )


@app.post("/create-sequence")
async def create_sequence(request: CreateSequenceRequest):
    """
    Create an episode sequence from multiple health/fitness/nutrition podcasts (URLs).
    """
    return await _create_sequence_internal(request.sources)


@app.post("/create-sequence-files")
async def create_sequence_files(files: List[UploadFile] = File(...)):
    """
    Create an episode sequence from multiple uploaded audio/video files.
    """
    if not files or len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 files are required for episode sequencing")
    
    temp_file_paths = []
    try:
        # Save each uploaded file temporarily
        for file in files:
            temp_file_path = TEMP_DIR / f"seq_upload_{file.filename}"
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_file_paths.append(str(temp_file_path))
        
        # Process the files
        return await _create_sequence_internal(temp_file_paths)
    finally:
        # Clean up temp files
        for path_str in temp_file_paths:
            path = Path(path_str)
            if path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete temp file {path}: {e}")
        

@app.get("/api/history")
async def history():
    """Get processing history."""
    try:
        return get_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
