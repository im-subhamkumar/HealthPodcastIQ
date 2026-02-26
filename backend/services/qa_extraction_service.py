"""Q&A extraction service using trained T5 model for health/fitness/nutrition podcasts."""
import re
import sys
from typing import List, Dict, Tuple

# Increase recursion limit for T5 model loading
# The transformers library's config parsing can hit default Python limits
sys.setrecursionlimit(5000)

# Try to import dependencies with helpful error messages
try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch not found. Please install it with: pip install torch"
    )

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError:
    raise ImportError(
        "Transformers library not found. Please install it with: pip install transformers"
    )

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Sentence-transformers not found. Please install it with: pip install sentence-transformers"
    )

from backend.config import QA_EXTRACTOR_MODEL_PATH, QA_PAIRS_PER_HOUR

# Global model instances (lazy loaded)
_qa_model = None
_qa_tokenizer = None
_embedding_model = None

# Comprehensive health/fitness/nutrition keywords for filtering
HEALTH_KEYWORDS = [
    # Core domains
    'health', 'fitness', 'nutrition', 'exercise', 'workout', 'diet', 'muscle',
    'training', 'cardio', 'strength', 'endurance', 'flexibility', 'mobility',
    
    # Macronutrients
    'protein', 'carbohydrate', 'carb', 'fat', 'calorie', 'calories', 'macronutrient',
    
    # Body composition
    'weight', 'loss', 'gain', 'metabolism', 'metabolic', 'body composition', 'lean mass',
    'fat loss', 'muscle growth', 'hypertrophy', 'atrophy',
    
    # Supplements & nutrients
    'supplement', 'vitamin', 'mineral', 'creatine', 'bcaa', 'omega', 'antioxidant',
    'electrolyte', 'hydration',
    
    # Recovery & performance
    'sleep', 'recovery', 'rest', 'overtraining', 'periodization', 'deload',
    
    # Hormones & physiology
    'hormone', 'cortisol', 'testosterone', 'insulin', 'glucose', 'ghrelin', 'leptin',
    'growth hormone', 'thyroid', 'adrenaline', 'noradrenaline',
    
    # Cardiovascular
    'blood', 'pressure', 'heart', 'rate', 'oxygen', 'vo2 max', 'cardiovascular',
    'aerobic', 'anaerobic', 'hiit', 'liss',
    
    # Cellular & molecular
    'mitochondria', 'mitochondrial', 'cellular', 'atp', 'glycolysis', 'oxidation',
    
    # Systems
    'immune', 'inflammation', 'inflammatory', 'immune system', 'autoimmune',
    
    # Mental & cognitive
    'stress', 'mental', 'cognitive', 'brain', 'neural', 'neuroplasticity', 'focus',
    'attention', 'memory', 'mood',
    
    # Physical health
    'bone', 'joint', 'injury', 'pain', 'inflammation', 'arthritis', 'tendon',
    'ligament', 'mobility', 'range of motion',
    
    # Research & science
    'study', 'research', 'evidence', 'clinical trial', 'meta-analysis', 'systematic review',
    'peer-reviewed', 'scientific', 'data', 'findings',
    
    # Common question words in health context
    'what', 'how', 'why', 'when', 'which', 'should', 'can', 'does', 'is', 'are',
    'effect', 'benefit', 'risk', 'safe', 'effective', 'recommend', 'suggest',
    'improve', 'increase', 'decrease', 'reduce', 'prevent', 'treat', 'cure',
    'cause', 'symptom', 'disease', 'condition', 'disorder', 'syndrome'
]


def get_qa_model():
    """Get or load Q&A model (singleton pattern)."""
    global _qa_model, _qa_tokenizer
    
    if _qa_model is None:
        print(f"Loading trained T5 Q&A extractor model from: {QA_EXTRACTOR_MODEL_PATH}")
        
        if not QA_EXTRACTOR_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Q&A extractor model not found at {QA_EXTRACTOR_MODEL_PATH}. "
                "Please ensure the model is trained and saved in the models directory."
            )
        
        try:
            # Check for required files
            required_files = ["config.json", "model.safetensors", "tokenizer_config.json"]
            missing_files = [f for f in required_files if not (QA_EXTRACTOR_MODEL_PATH / f).exists()]
            if missing_files:
                raise FileNotFoundError(
                    f"Missing required model files: {', '.join(missing_files)}. "
                    f"Model directory: {QA_EXTRACTOR_MODEL_PATH}"
                )
            
            print(f"  Loading tokenizer from {QA_EXTRACTOR_MODEL_PATH}...")
            _qa_tokenizer = T5Tokenizer.from_pretrained(str(QA_EXTRACTOR_MODEL_PATH))
            
            print(f"  Loading model from {QA_EXTRACTOR_MODEL_PATH}...")
            _qa_model = T5ForConditionalGeneration.from_pretrained(str(QA_EXTRACTOR_MODEL_PATH))
            
            # Set to evaluation mode
            _qa_model.eval()
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _qa_model.to(device)
            print(f"✓ Trained T5 Q&A extractor model loaded successfully on {device}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"QA extractor model files not found: {str(e)}. "
                f"Please ensure the model is properly installed at {QA_EXTRACTOR_MODEL_PATH}"
            )
        except OSError as e:
            raise RuntimeError(
                f"Failed to load QA extractor model (OS error): {str(e)}. "
                f"This may indicate corrupted model files or missing dependencies."
            )
        except Exception as e:
            error_type = type(e).__name__
            raise RuntimeError(
                f"Failed to load QA extractor model ({error_type}): {str(e)}. "
                f"Model path: {QA_EXTRACTOR_MODEL_PATH}"
            )
    
    return _qa_model, _qa_tokenizer


def get_embedding_model():
    """Get or load sentence transformer model for semantic search."""
    global _embedding_model
    
    if _embedding_model is None:
        print("Loading sentence transformer model for answer extraction...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence transformer model loaded")
    
    return _embedding_model


def chunk_transcript(transcript: str, chunk_size: int = 600, overlap: int = 100, _recursion_depth: int = 0) -> List[str]:
    """
    Split transcript into overlapping chunks for Q&A generation.
    
    Args:
        transcript: Full transcript text
        chunk_size: Target chunk size in words
        overlap: Overlap between chunks in words
        _recursion_depth: Internal recursion tracker to prevent infinite loops
        
    Returns:
        List of transcript chunks
    """
    # Prevent infinite recursion
    if _recursion_depth >= 2:
        # Return whatever we can with current parameters
        words = transcript.split()
        if len(words) < 50:
            return [transcript] if transcript.strip() else []
        # Just create chunks without further recursion
        chunks = []
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip() and len(chunk.strip()) > 50:
                chunks.append(chunk)
            i += max(chunk_size - overlap, 50)  # Ensure we always move forward
        return chunks if chunks else [transcript] if transcript.strip() else []
    
    words = transcript.split()
    chunks = []
    
    if len(words) < chunk_size:
        # If transcript is shorter than chunk size, return as single chunk
        return [transcript] if transcript.strip() else []
    
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip() and len(chunk.strip()) > 50:  # Ensure meaningful chunks
            chunks.append(chunk)
        i += chunk_size - overlap
        
        if i >= len(words):
            break
    
    # Ensure we have enough chunks for target Q&A pairs (need more chunks than target)
    min_chunks_needed = 15
    if len(chunks) < min_chunks_needed and len(words) > chunk_size:
        # Re-chunk with smaller size to get more chunks
        smaller_chunk_size = max(100, len(words) // min_chunks_needed)
        # Only recurse if the new size is meaningfully smaller
        if smaller_chunk_size < chunk_size * 0.7:
            return chunk_transcript(transcript, chunk_size=smaller_chunk_size, overlap=50, _recursion_depth=_recursion_depth + 1)
    
    return chunks



def generate_question(context: str) -> str:
    """
    Generate a health/fitness/nutrition question from context using trained T5 model.
    
    Args:
        context: Context text from health/fitness/nutrition podcast
        
    Returns:
        Generated question focused on health/fitness/nutrition
    """
    try:
        model, tokenizer = get_qa_model()
        
        # Format input as per training - the model was trained on health/fitness content
        # Use prefix that matches training format
        input_text = f"generate a health claim question: {context}"
        
        # Truncate context if too long (keep last part which is usually more relevant)
        if len(input_text) > 500:
            # Keep last 400 characters of context
            context_words = context.split()
            context = ' '.join(context_words[-200:])  # Last 200 words
            input_text = f"generate a health claim question: {context}"
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            max_length=512,  # T5 max input length
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate question with parameters optimized for health Q&A
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,  # Increased from 80
                min_length=10,  # Reduced from 15
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Reduced from 1.3
                length_penalty=1.1,  # Reduced from 1.2
                do_sample=False  # Deterministic generation
            )
        
        # Decode output
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        question = question.strip()
        
        # Clean up question
        if not question:
            return ""
        
        # Remove any trailing punctuation artifacts
        question = re.sub(r'[.]+$', '', question)
        if not question.endswith('?'):
            question += '?'
        
        return question
        
    except Exception as e:
        print(f"Error in generate_question: {e}")
        import traceback
        traceback.print_exc()
        # Return a simple question as fallback
        return "What is discussed in this content?"


def is_health_related(text: str) -> bool:
    """
    Check if text is related to health/fitness/nutrition.
    Uses comprehensive keyword matching and context analysis.
    
    Args:
        text: Text to check
        
    Returns:
        True if health-related
    """
    if not text or len(text.strip()) < 10:
        return False
    
    text_lower = text.lower()
    
    # Count keyword matches (check both whole word and substring for better matching)
    matches = 0
    for keyword in HEALTH_KEYWORDS:
        # Check if keyword appears in text
        if keyword in text_lower:
            matches += 1
            # Give extra weight for whole word matches
            if f' {keyword} ' in text_lower or text_lower.startswith(keyword + ' ') or text_lower.endswith(' ' + keyword):
                matches += 0.5  # Bonus for whole word match
    
    # Require at least 1 keyword match
    # This ensures we capture health-related content while being lenient enough
    return matches >= 1


def extract_answer(question: str, transcript: str) -> str:
    """
    Extract answer from transcript using semantic search.
    Ensures answer comes from transcript (no hallucinations).
    
    Args:
        question: Generated question
        transcript: Full transcript text
        
    Returns:
        Most relevant answer segment from transcript (verified to be in transcript)
    """
    from sentence_transformers import util as st_util
    import numpy as np
    
    embedding_model = get_embedding_model()
    
    # Split transcript into sentences with better handling
    # Use more sophisticated sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]  # Filter very short sentences
    
    if not sentences:
        # If no good sentences, try splitting by periods only
        sentences = [s.strip() for s in transcript.split('.') if len(s.strip()) > 20]
    
    if not sentences:
        return "Answer not found in transcript."
    
    # Handle single sentence case
    if len(sentences) == 1:
        return sentences[0]
    
    try:
        # Get embeddings
        question_embedding = embedding_model.encode(question, convert_to_tensor=True)
        sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
        
        # Use sentence-transformers util for proper cosine similarity
        # This handles tensor dimensions correctly
        similarities = st_util.cos_sim(question_embedding, sentence_embeddings)[0]
        
        # Convert to numpy for safe sorting
        similarities_np = similarities.cpu().numpy()
        
        # Get top 3-5 most similar sentences
        similarity_threshold = 0.2
        top_k = min(5, len(sentences))
        
        # Sort indices by similarity (descending)
        sorted_indices = np.argsort(similarities_np)[::-1]
        top_indices = sorted_indices[:top_k]
        
        # Filter by similarity threshold and select best matches
        relevant_sentences = []
        for idx in top_indices:
            if similarities_np[idx] >= similarity_threshold:
                relevant_sentences.append(sentences[idx])
        
        if not relevant_sentences:
            # If no high-similarity matches, use top 2-3 anyway (always return something)
            relevant_sentences = [sentences[idx] for idx in top_indices[:min(3, len(top_indices))]]
        
        # Combine into coherent answer
        answer = ' '.join(relevant_sentences)
        
        # Verify answer is actually in transcript (anti-hallucination check)
        answer_normalized = answer.lower().strip()
        transcript_normalized = transcript.lower()
        
        # Check if key phrases from answer exist in transcript
        answer_words = set(answer_normalized.split())
        if len(answer_words) < 3:
            return sentences[top_indices[0]] if len(top_indices) > 0 else "Answer not found in transcript."
        
        # Ensure at least 50% of answer words appear in transcript
        matching_words = sum(1 for word in answer_words if word in transcript_normalized)
        if matching_words / len(answer_words) < 0.5:
            # Answer might be hallucinated, use top sentence only
            if relevant_sentences:
                answer = relevant_sentences[0]
            elif len(top_indices) > 0:
                answer = sentences[top_indices[0]]
            else:
                return "Answer not found in transcript."
        
        return answer.strip()
        
    except Exception as e:
        print(f"Error in extract_answer: {e}")
        # Fallback: return first relevant sentence
        if sentences:
            return sentences[0]
        return "Answer not found in transcript."


def generate_answer_versions(answer: str) -> Dict[str, str]:
    """
    Generate short, medium, and long versions of answer.
    
    Args:
        answer: Base answer text
        
    Returns:
        Dictionary with "short", "medium", "long" answer versions
    """
    sentences = answer.split('. ')
    
    short = sentences[0] if sentences else answer
    if len(short) > 150:
        short = short[:150] + "..."
    
    medium = '. '.join(sentences[:2]) if len(sentences) >= 2 else answer
    if len(medium) > 300:
        medium = medium[:300] + "..."
    
    long = answer
    
    return {
        "short": short,
        "medium": medium,
        "long": long
    }


def extract_qa_pairs(transcript: str, estimated_duration_hours: float = 1.0) -> List[Dict]:
    """
    Extract health/fitness/nutrition Q&A pairs from transcript using trained T5 model.
    Strictly focuses on health-related content and ensures no hallucinations.
    
    Args:
        transcript: Full transcript text from health/fitness/nutrition podcast
        estimated_duration_hours: Estimated duration in hours (for determining number of Q&A pairs)
        
    Returns:
        List of Q&A pair dictionaries with verified answers from transcript
    """
    print("Extracting health/fitness/nutrition Q&A pairs using trained T5 model...")
    
    # Calculate target number of Q&A pairs (minimum 10, more for longer content)
    target_pairs = max(10, int(estimated_duration_hours * QA_PAIRS_PER_HOUR))
    print(f"Target: {target_pairs} health/fitness/nutrition Q&A pairs")
    
    # Chunk transcript for question generation
    # Use smaller chunks with good overlap to maximize coverage and get more Q&A pairs
    # Strategy: Create enough chunks to ensure we can find health content
    chunk_size = 250  # Smaller chunks = more questions
    overlap = 80
    chunks = chunk_transcript(transcript, chunk_size=chunk_size, overlap=overlap)
    print(f"Split transcript into {len(chunks)} chunks for processing")
    
    # Ensure we have enough chunks (need at least 2x target_pairs to find health content)
    min_chunks_needed = max(target_pairs * 2, 20)  # At least 20 chunks
    if len(chunks) < min_chunks_needed:
        # Use even smaller chunks to get more coverage
        smaller_size = max(150, len(transcript.split()) // min_chunks_needed)
        chunks = chunk_transcript(transcript, chunk_size=smaller_size, overlap=50)
        print(f"Regenerated with {len(chunks)} smaller chunks (target: {min_chunks_needed}) to ensure enough coverage")
    
    qa_pairs = []
    seen_questions = set()
    # Process ALL chunks to find health-related content
    max_attempts = len(chunks) * 2  # Process all chunks, allow retries
    attempts = 0
    processed_chunks = set()
    
    # First pass: STRICTLY get health-related Q&A pairs only
    print(f"Processing {len(chunks)} chunks to find health/fitness/nutrition content...")
    health_chunks_found = 0
    
    # Process ALL health-related chunks to maximize Q&A pairs
    for i, chunk in enumerate(chunks):
        # Process ALL chunks to maximize health Q&A pairs
        # Only stop if we have significantly more than target (150%) OR processed 95% of chunks
        if len(qa_pairs) >= target_pairs * 1.5:
            print(f"Generated {len(qa_pairs)} Q&A pairs (150% of target). Stopping.")
            break
        if len(qa_pairs) >= target_pairs and i >= len(chunks) * 0.95:
            # Processed 95% of chunks and have target pairs, can stop
            print(f"Reached target of {target_pairs} Q&A pairs after processing {i+1}/{len(chunks)} chunks (95%)")
            break
        
        # Skip if chunk is too short (need enough context)
        if len(chunk.split()) < 40:
            continue
        
        # STRICT: Only process health-related chunks
        if not is_health_related(chunk):
            if i < 10:  # Log first few non-health chunks
                print(f"  Chunk {i}: Not health-related, skipping")
            continue
        
        health_chunks_found += 1
        if health_chunks_found <= 5:
            print(f"  ✓ Chunk {i}: Health-related content found ({health_chunks_found} health chunks so far)")
        
        # Skip if we've already processed this chunk
        chunk_hash = hash(chunk[:100])  # Use hash to identify similar chunks
        if chunk_hash in processed_chunks:
            continue
        processed_chunks.add(chunk_hash)
        
        attempts += 1
        
        # Generate question using trained model
        try:
            question = generate_question(chunk)
            
            # Debug logging for first few attempts
            if i < 3:
                print(f"  Chunk {i}: Generated question: {question[:80] if question else 'None'}...")
            
            # More lenient filtering for questions
            if not question or len(question.strip()) < 8:  # Very lenient minimum
                if i < 3:
                    print(f"  Chunk {i}: Question too short or empty, skipping")
                continue
            
            if len(question) > 250:  # Increased from 200
                if i < 3:
                    print(f"  Chunk {i}: Question too long, skipping")
                continue
            
            if question.lower() in seen_questions:
                if i < 3:
                    print(f"  Chunk {i}: Duplicate question, skipping")
                continue
            
            # STRICT: Question MUST be health-related
            if not is_health_related(question):
                if i < 5:  # Log more attempts
                    print(f"  Chunk {i}: Question not health/fitness/nutrition related, skipping: {question[:60]}...")
                continue
            
            seen_questions.add(question.lower())
            
            # Extract answer from full transcript (ensures no hallucination)
            answer = extract_answer(question, transcript)
            
            # More lenient validation of answer
            if (len(answer) < 15 or  # Reduced from 20
                "not found" in answer.lower()):
                if i < 3:
                    print(f"  Chunk {i}: Answer validation failed (length: {len(answer)})")
                continue
            
            # Don't require answer to be health-related if question is health-related
            # (answers might not contain keywords but still be relevant)
            
            # Generate answer versions (short, medium, long)
            answer_versions = generate_answer_versions(answer)
            
            # Validate answer versions (more lenient)
            if not all(len(v) > 5 for v in answer_versions.values()):  # Reduced from 10
                continue
            
            # Create Q&A pair
            qa_pair = {
                "id": f"qa{len(qa_pairs) + 1}",
                "question": question,
                "answers": answer_versions,
                "claims": []  # Will be populated by fact-checking service
            }
            
            qa_pairs.append(qa_pair)
            print(f"✓ Generated Q&A pair {len(qa_pairs)}/{target_pairs}: {question[:70]}...")
            
            # If we're making good progress, continue
            if len(qa_pairs) % 5 == 0:
                print(f"  Progress: {len(qa_pairs)}/{target_pairs} health Q&A pairs generated")
            
        except Exception as e:
            print(f"Error generating Q&A from chunk {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"First pass complete: Found {health_chunks_found} health-related chunks, generated {len(qa_pairs)} Q&A pairs")
    
    # Second pass: If we didn't get enough health Q&A pairs, process remaining chunks
    if len(qa_pairs) < target_pairs:
        print(f"Need {target_pairs - len(qa_pairs)} more health Q&A pairs. Processing remaining chunks...")
        remaining_chunks = [c for i, c in enumerate(chunks) if hash(c[:100]) not in processed_chunks]
        print(f"Processing {len(remaining_chunks)} remaining chunks for health content...")
        
        health_chunks_remaining = sum(1 for c in remaining_chunks if is_health_related(c))
        print(f"Found {health_chunks_remaining} health-related chunks in remaining set")
        
        for i, chunk in enumerate(remaining_chunks):
            if len(qa_pairs) >= target_pairs:
                break
            
            # Still require health-related content
            if len(chunk.split()) < 40:
                continue
            
            if not is_health_related(chunk):
                continue
            
            chunk_hash = hash(chunk[:100])
            if chunk_hash in processed_chunks:
                continue
            processed_chunks.add(chunk_hash)
            
            try:
                question = generate_question(chunk)
                
                # STRICT: Must be health-related
                if not question or len(question.strip()) < 8:
                    continue
                
                if question.lower() in seen_questions:
                    continue
                
                if not is_health_related(question):
                    continue
                
                seen_questions.add(question.lower())
                answer = extract_answer(question, transcript)
                
                if len(answer) < 15 or "not found" in answer.lower():
                    continue
                
                answer_versions = generate_answer_versions(answer)
                if not all(len(v) > 5 for v in answer_versions.values()):
                    continue
                
                qa_pair = {
                    "id": f"qa{len(qa_pairs) + 1}",
                    "question": question,
                    "answers": answer_versions,
                    "claims": []
                }
                
                qa_pairs.append(qa_pair)
                print(f"✓ Second pass: Generated Q&A pair {len(qa_pairs)}/{target_pairs}: {question[:70]}...")
            except Exception as e:
                print(f"Error in second pass chunk {i}: {e}")
                continue
    
    # Final check: If we still don't have enough, try one more aggressive pass
    if len(qa_pairs) < target_pairs:
        print(f"Still need {target_pairs - len(qa_pairs)} more Q&A pairs. Trying aggressive extraction...")
        
        # Try processing chunks with even smaller size to get more coverage
        small_chunks = chunk_transcript(transcript, chunk_size=200, overlap=50)
        print(f"Trying {len(small_chunks)} smaller chunks...")
        
        for i, chunk in enumerate(small_chunks):
            if len(qa_pairs) >= target_pairs:
                break
            
            if len(chunk.split()) < 30:
                continue
            
            # Must be health-related
            if not is_health_related(chunk):
                continue
            
            chunk_hash = hash(chunk[:100])
            if chunk_hash in processed_chunks:
                continue
            processed_chunks.add(chunk_hash)
            
            try:
                question = generate_question(chunk)
                
                if not question or len(question.strip()) < 8:
                    continue
                
                if question.lower() in seen_questions:
                    continue
                
                # STRICT: Must be health-related
                if not is_health_related(question):
                    continue
                
                seen_questions.add(question.lower())
                answer = extract_answer(question, transcript)
                
                if len(answer) < 15 or "not found" in answer.lower():
                    continue
                
                answer_versions = generate_answer_versions(answer)
                if not all(len(v) > 5 for v in answer_versions.values()):
                    continue
                
                qa_pair = {
                    "id": f"qa{len(qa_pairs) + 1}",
                    "question": question,
                    "answers": answer_versions,
                    "claims": []
                }
                
                qa_pairs.append(qa_pair)
                print(f"✓ Aggressive pass: Generated Q&A pair {len(qa_pairs)}/{target_pairs}: {question[:70]}...")
            except Exception as e:
                continue
    
    # Final fallback: Only if we have ZERO pairs (shouldn't happen for health content)
    if len(qa_pairs) == 0:
        print("⚠️  Warning: No health Q&A pairs generated. This may indicate the content is not health/fitness/nutrition related.")
        print("Creating minimal fallback Q&A pair...")
        # Create at least one Q&A pair from the transcript
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 50]
        
        if sentences:
            # Use first substantial sentence as answer
            answer_text = sentences[0] if len(sentences[0]) > 100 else ' '.join(sentences[:2])
            question = "What health or fitness topics are discussed in this podcast?"
            
            qa_pair = {
                "id": "qa1",
                "question": question,
                "answers": generate_answer_versions(answer_text),
                "claims": []
            }
            qa_pairs.append(qa_pair)
            print("Created minimal fallback Q&A pair")
    elif len(qa_pairs) < target_pairs:
        print(f"⚠️  Warning: Only generated {len(qa_pairs)}/{target_pairs} health Q&A pairs. Content may not be fully health/fitness/nutrition focused.")
    
    print(f"✓ Successfully generated {len(qa_pairs)} health/fitness/nutrition Q&A pairs")
    return qa_pairs

