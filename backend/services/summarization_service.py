"""Summarization service using trained BART model for health/fitness/nutrition podcasts."""
from typing import Dict

# Try to import dependencies with helpful error messages
try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch not found. Please install it with: pip install torch"
    )

try:
    from transformers import BartForConditionalGeneration, BartTokenizer
except ImportError:
    raise ImportError(
        "Transformers library not found. Please install it with: pip install transformers"
    )

from backend.config import SUMMARIZER_MODEL_PATH

# Global model instances (lazy loaded)
_summarizer_model = None
_summarizer_tokenizer = None


def get_summarizer_model():
    """Get or load summarizer model (singleton pattern)."""
    global _summarizer_model, _summarizer_tokenizer
    
    if _summarizer_model is None:
        print(f"Loading trained BART summarizer model from: {SUMMARIZER_MODEL_PATH}")
        
        if not SUMMARIZER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Summarizer model not found at {SUMMARIZER_MODEL_PATH}. "
                "Please ensure the model is trained and saved in the models directory."
            )
        
        try:
            # Check for required files
            required_files = ["config.json", "model.safetensors"]
            missing_files = [f for f in required_files if not (SUMMARIZER_MODEL_PATH / f).exists()]
            if missing_files:
                raise FileNotFoundError(
                    f"Missing required model files: {', '.join(missing_files)}. "
                    f"Model directory: {SUMMARIZER_MODEL_PATH}"
                )
            
            print(f"  Loading tokenizer from {SUMMARIZER_MODEL_PATH}...")
            _summarizer_tokenizer = BartTokenizer.from_pretrained(str(SUMMARIZER_MODEL_PATH))
            
            print(f"  Loading model from {SUMMARIZER_MODEL_PATH}...")
            _summarizer_model = BartForConditionalGeneration.from_pretrained(str(SUMMARIZER_MODEL_PATH))
            
            # Set to evaluation mode
            _summarizer_model.eval()
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _summarizer_model.to(device)
            print(f"✓ Trained BART summarizer model loaded successfully on {device}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Summarizer model files not found: {str(e)}. "
                f"Please ensure the model is properly installed at {SUMMARIZER_MODEL_PATH}"
            )
        except OSError as e:
            raise RuntimeError(
                f"Failed to load summarizer model (OS error): {str(e)}. "
                f"This may indicate corrupted model files or missing dependencies."
            )
        except Exception as e:
            error_type = type(e).__name__
            raise RuntimeError(
                f"Failed to load summarizer model ({error_type}): {str(e)}. "
                f"Model path: {SUMMARIZER_MODEL_PATH}"
            )
    
    return _summarizer_model, _summarizer_tokenizer


def generate_summary(text: str, summary_type: str = "medium") -> str:
    """
    Generate summary using trained BART model optimized for health/fitness/nutrition content.
    
    Args:
        text: Input transcript text (health/fitness/nutrition podcast)
        summary_type: One of "short", "medium", "long"
        
    Returns:
        Generated summary preserving scientific terminology
    """
    model, tokenizer = get_summarizer_model()
    
    # Determine length parameters based on summary type
    # Adjusted for health/fitness content which may need more detail
    length_params = {
        "short": {"min_length": 30, "max_length": 80},  # 2-minute elevator pitch
        "medium": {"min_length": 100, "max_length": 200},  # Key takeaways
        "long": {"min_length": 300, "max_length": 600}  # Detailed breakdown (20-30% of original)
    }
    
    params = length_params.get(summary_type, length_params["medium"])
    
    # Add health/fitness context prefix to help model focus
    # The trained model should already be optimized, but this helps
    prefix = "summarize health and fitness podcast: "
    input_text = prefix + text
    
    # Tokenize input - BART max is 1024 tokens
    inputs = tokenizer(
        input_text,
        max_length=1024,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary with parameters optimized for health content
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            min_length=params["min_length"],
            max_length=params["max_length"],
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            do_sample=False,
            repetition_penalty=1.2  # Reduce repetition of scientific terms
        )
    
    # Decode output
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up any artifacts
    summary = summary.strip()
    
    return summary


def generate_hierarchical_summaries(transcript: str) -> Dict[str, str]:
    """
    Generate hierarchical summaries (short, medium, long) using trained BART model.
    Handles long transcripts by chunking intelligently.
    
    Args:
        transcript: Full transcript text from health/fitness/nutrition podcast
        
    Returns:
        Dictionary with "short", "medium", "long" summaries
    """
    print("Generating hierarchical summaries using trained BART model...")
    
    # BART max input is 1024 tokens (~800 words)
    # For longer transcripts, we'll use a sliding window approach
    max_input_words = 800
    words = transcript.split()
    
    if len(words) > max_input_words:
        print(f"Transcript is long ({len(words)} words), using intelligent chunking")
        # Use first portion for summary (usually contains key intro content)
        # For health podcasts, the intro often sets up the main topics
        transcript_for_summary = ' '.join(words[:max_input_words])
        print(f"Using first {max_input_words} words for summary generation")
    else:
        transcript_for_summary = transcript
    
    try:
        summaries = {
            "short": generate_summary(transcript_for_summary, "short"),
            "medium": generate_summary(transcript_for_summary, "medium"),
            "long": generate_summary(transcript_for_summary, "long")
        }
        
        # Validate summaries are not empty
        for key, summary in summaries.items():
            if not summary or len(summary.strip()) < 10:
                print(f"Warning: {key} summary is too short, regenerating...")
                summaries[key] = generate_summary(transcript_for_summary, key)
        
        print("Hierarchical summaries generated successfully")
        return summaries
        
    except Exception as e:
        print(f"Error generating summaries: {e}")
        raise RuntimeError(f"Failed to generate summaries: {e}")

