"""Fact-checking service with internal transcript verification and Bytez/Gemini API verification."""
import re
import os
import requests
from typing import List, Dict, Tuple

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Sentence-transformers not found. Please install it with: pip install sentence-transformers"
    )

from backend.config import GEMINI_API_KEY, BYTEZ_API_KEY, SAMBANOVA_API_KEY

# Determine APIs to use
USE_BYTEZ = bool(BYTEZ_API_KEY)
USE_SAMBANOVA = bool(SAMBANOVA_API_KEY)

if USE_BYTEZ:
    print("Bytez API configured for fact-checking")
if USE_SAMBANOVA:
    print("SambaNova API configured for fact-checking (DeepSeek-R1-Distill-Llama-70B)")

if not USE_BYTEZ and not USE_SAMBANOVA:
    print("Warning: No API keys configured (BYTEZ_API_KEY or SAMBANOVA_API_KEY). External fact-checking will be disabled.")


# Global embedding model for internal verification
_embedding_model = None


def get_embedding_model():
    """Get or load sentence transformer model for internal verification."""
    global _embedding_model
    
    if _embedding_model is None:
        print("Loading sentence transformer for internal fact-checking...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence transformer loaded")
    
    return _embedding_model


def extract_claims_from_answer(answer: str) -> List[str]:
    """
    Extract factual health/fitness/nutrition claims from an answer.
    Focuses on verifiable scientific claims.
    
    Args:
        answer: Answer text to extract claims from
        
    Returns:
        List of extracted verifiable claims (max 2)
    """
    claims = []
    
    # Pattern 1: Statements with specific numbers/quantities (most verifiable)
    number_pattern = r'[A-Z][^.!?]*(?:\d+[\.\-]?\d*\s*(?:to|-|and)\s*\d+[\.\-]?\d*|\d+[%]?|one|two|three|four|five|six|seven|eight|nine|ten)\s+[^.!?]*[.!?]'
    number_claims = re.findall(number_pattern, answer, re.IGNORECASE)
    claims.extend([c.strip() for c in number_claims if len(c.strip()) > 25 and len(c.strip()) < 200])
    
    # Pattern 2: Research-backed statements
    research_pattern = r'[A-Z][^.!?]*(?:studies?|research|evidence|data|findings?|meta-analysis|systematic review|clinical trial|peer-reviewed)\s+(?:show|indicate|suggest|demonstrate|find|confirm|prove)[^.!?]*[.!?]'
    research_claims = re.findall(research_pattern, answer, re.IGNORECASE)
    claims.extend([c.strip() for c in research_claims if len(c.strip()) > 30 and len(c.strip()) < 200])
    
    # Pattern 3: Causal health claims
    health_mechanism_pattern = r'[A-Z][^.!?]*(?:improves?|increases?|decreases?|reduces?|enhances?|boosts?|affects?|impacts?|influences?|regulates?)\s+[^.!?]*(?:health|fitness|muscle|metabolism|hormone|immune|brain|heart|performance|recovery|sleep|weight|fat|protein|carbohydrate|glucose|insulin|cortisol|testosterone)[^.!?]*[.!?]'
    health_claims = re.findall(health_mechanism_pattern, answer, re.IGNORECASE)
    claims.extend([c.strip() for c in health_claims if len(c.strip()) > 30 and len(c.strip()) < 200])
    
    # Pattern 4: Comparative statements
    comparative_pattern = r'[A-Z][^.!?]*(?:better|more|less|superior|inferior|effective|efficient)\s+[^.!?]*(?:than|compared to|versus)[^.!?]*[.!?]'
    comparative_claims = re.findall(comparative_pattern, answer, re.IGNORECASE)
    claims.extend([c.strip() for c in comparative_claims if len(c.strip()) > 25 and len(c.strip()) < 200])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_claims = []
    for claim in claims:
        claim_normalized = re.sub(r'\s+', ' ', claim.lower().strip())
        if claim_normalized not in seen and len(claim) > 20:
            seen.add(claim_normalized)
            unique_claims.append(claim)
    
    # Filter out very generic claims
    generic_phrases = ['it is', 'this is', 'that is', 'there is', 'there are']
    filtered_claims = [
        c for c in unique_claims 
        if not any(c.lower().startswith(phrase) for phrase in generic_phrases)
    ]
    
    # Prioritize claims with numbers or research references
    prioritized = sorted(
        filtered_claims,
        key=lambda x: (
            bool(re.search(r'\d+', x)),
            bool(re.search(r'(?:study|research|evidence)', x, re.I)),
            len(x)
        ),
        reverse=True
    )
    
    return prioritized[:2]


def verify_claim_internal(claim: str, transcript: str) -> Tuple[str, float]:
    """
    Verify claim against transcript using semantic similarity.
    
    Args:
        claim: Claim to verify
        transcript: Full transcript text
        
    Returns:
        Tuple of (verdict, confidence_score)
    """
    from sentence_transformers import util as st_util
    import numpy as np
    
    embedding_model = get_embedding_model()
    
    # Split transcript into sentences
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return "NEUTRAL", 0.0
    
    try:
        # Get embeddings
        claim_embedding = embedding_model.encode(claim, convert_to_tensor=True)
        sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
        
        # Calculate cosine similarity using sentence-transformers util
        similarities = st_util.cos_sim(claim_embedding, sentence_embeddings)[0]
        
        # Get max similarity
        max_similarity = float(similarities.max().item())
        
        if max_similarity > 0.75:
            confidence = min(95, int(max_similarity * 100))
            return "SUPPORTS", confidence
        elif max_similarity > 0.50:
            confidence = int(max_similarity * 80)
            return "NEUTRAL", confidence
        else:
            return "NEUTRAL", 30
    except Exception as e:
        print(f"Error in internal verification: {e}")
        return "NEUTRAL", 0.0


def verify_claim_with_bytez(claim: str, internal_verdict: str = "NEUTRAL") -> Dict:
    """
    Verify a health/fitness/nutrition claim using Bytez API with retry logic.
    
    Args:
        claim: Claim to verify
        internal_verdict: Result from internal transcript verification
        
    Returns:
        Dictionary with verdict, confidence, and explanation
    """
    import time
    if not BYTEZ_API_KEY:
        return {
            "claim": claim,
            "verdict": "NEUTRAL",
            "confidence": 0,
            "explanation": "External fact-checking unavailable: BYTEZ_API_KEY not configured."
        }
    
    max_retries = 3
    retry_delay = 2  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            # Bytez API endpoint (OpenAI compatible)
            url = "https://api.bytez.com/models/v2/openai/v1/chat/completions"
            
            headers = {
                "Authorization": BYTEZ_API_KEY,
                "Content-Type": "application/json"
            }
            
            # Create verification prompt
            prompt = f"""You are an expert scientific fact-checker specializing in health, fitness, and nutrition research. Verify the following claim against peer-reviewed scientific literature.
            
CLAIM TO VERIFY: "{claim}"

INTERNAL TRANSCRIPT CHECK: {internal_verdict}

Provide your assessment in the following format:

1. VERDICT: Choose ONE - SUPPORTS (claim is scientifically supported), REFUTES (contradicts evidence), or NEUTRAL (mixed/insufficient evidence).

2. CONFIDENCE: Provide a number from 0 to 100 based on the quality and strength of peer-reviewed scientific evidence. Be precise (e.g., don't just use 50).

3. EXPLANATION: Provide exactly 3 to 4 concise sentences. YOU MUST MENTION AT LEAST ONE AUTHORITATIVE SOURCE (e.g., NIH, Mayo Clinic, PubMed, etc.). Focus on the core scientific truth.

Format EXACTLY as:
VERDICT: [SUPPORTS/REFUTES/NEUTRAL]
CONFIDENCE: [0-100]
EXPLANATION: [your concise 3-4 line explanation with sources]"""
            
            payload = {
                "model": "Qwen/Qwen3-1.7B",  # Using Qwen model available on Bytez
                "messages": [
                    {"role": "system", "content": "You are an expert scientific fact-checker for health/fitness/nutrition claims."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.3
            }
            
            # Increased timeout to 60 seconds to prevent ReadTimeout
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 429:
                # Rate limit hit, wait and retry
                print(f"  Rate limit hit (429), retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
                
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Parse response
            verdict = "NEUTRAL"
            confidence = 0 # Default to 0 if parsing fails
            explanation = response_text
            
            # Extract verdict - more robust regex
            verdict_match = re.search(r'VERDICT:\s*\*?\*?\s*(SUPPORTS|REFUTES|NEUTRAL)', response_text, re.IGNORECASE)
            if verdict_match:
                verdict = verdict_match.group(1).upper()
            
            # Extract confidence - more robust regex
            confidence_match = re.search(r'CONFIDENCE:\s*\*?\*?\s*(\d+)', response_text)
            if confidence_match:
                confidence = max(0, min(100, int(confidence_match.group(1))))
            else:
                # Fallback: if we found a number near "confidence" but not with colon
                num_match = re.search(r'confidence.*?(\d+)', response_text, re.IGNORECASE | re.DOTALL)
                if num_match:
                    confidence = max(0, min(100, int(num_match.group(1))))
                else:
                    # If we can't find it, use a heuristic based on verdict
                    if verdict == "SUPPORTS": confidence = 70
                    elif verdict == "REFUTES": confidence = 85
                    else: confidence = 30
            
            # Extract explanation
            explanation_match = re.search(r'EXPLANATION:\s*\*?\*?\s*(.+)', response_text, re.DOTALL | re.IGNORECASE)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                # Ensure it's not too long
                lines = explanation.split('.')
                if len(lines) > 5:
                    explanation = '.'.join(lines[:4]).strip() + '.'
            
            # Boost confidence slightly if internal and external agree
            if internal_verdict == "SUPPORTS" and verdict == "SUPPORTS":
                confidence = min(98, confidence + 5)
            
            if not confidence_match:
                print(f"  Warning: Could not parse confidence from Bytez response. Defaulted to {confidence}.")
                print(f"  Full response: {response_text[:200]}...")
            
            return {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "explanation": explanation
            }
            
        except requests.exceptions.ReadTimeout:
            print(f"  Bytez API timeout, retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2
            continue
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error verifying claim with Bytez (attempt {attempt + 1}): {e}. Retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            
            print(f"Error verifying claim with Bytez after {max_retries} attempts: {e}")
            
            # Final fallback to Gemini if Bytez failed completely
            if GEMINI_API_KEY:
                print("  Attempting fallback to Gemini API...")
                return verify_claim_with_gemini(claim, internal_verdict)
                
            return {
                "claim": claim,
                "verdict": "NEUTRAL",
                "confidence": 0,
                "explanation": f"Error during external fact-checking: {str(e)}"
            }
    
    # If all retries failed and no return happened
    return {
        "claim": claim,
        "verdict": "NEUTRAL",
        "confidence": 0,
        "explanation": "External fact-checking failed after multiple attempts."
    }



def verify_claim_with_sambanova(claim: str, internal_verdict: str = "NEUTRAL") -> Dict:
    """
    Verify a health/fitness/nutrition claim using SambaNova API (DeepSeek-R1-Distill-Llama-70B).
    
    Args:
        claim: Claim to verify
        internal_verdict: Result from internal transcript verification
        
    Returns:
        Dictionary with verdict, confidence, and explanation
    """
    import time
    if not SAMBANOVA_API_KEY:
        return {
            "claim": claim,
            "verdict": "NEUTRAL",
            "confidence": 0,
            "explanation": "External fact-checking unavailable: SAMBANOVA_API_KEY not configured."
        }
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # SambaNova API endpoint (OpenAI compatible)
            url = "https://api.sambanova.ai/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Create verification prompt
            prompt = f"""You are an expert scientific fact-checker specializing in health, fitness, and nutrition research. Verify the following claim against peer-reviewed scientific literature.

CLAIM TO VERIFY: "{claim}"

INTERNAL TRANSCRIPT CHECK: {internal_verdict}

Provide your assessment in the following format:

1. VERDICT: Choose ONE - SUPPORTS (claim is scientifically supported), REFUTES (contradicts evidence), or NEUTRAL (mixed/insufficient evidence).

2. CONFIDENCE: Provide a number from 0 to 100 based on the quality and strength of peer-reviewed scientific evidence. Be precise (e.g., don't just use 50).

3. EXPLANATION: Provide exactly 3 to 4 concise sentences. YOU MUST MENTION AT LEAST ONE AUTHORITATIVE SOURCE (e.g., NIH, Mayo Clinic, PubMed, etc.). Focus on the core scientific truth.

Format EXACTLY as:
VERDICT: [SUPPORTS/REFUTES/NEUTRAL]
CONFIDENCE: [0-100]
EXPLANATION: [your concise 3-4 line explanation with sources]"""
            
            payload = {
                "model": "DeepSeek-R1-Distill-Llama-70B",
                "messages": [
                    {"role": "system", "content": "You are an expert scientific fact-checker for health/fitness/nutrition claims."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 429:
                print(f"  SambaNova Rate limit hit (429), retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
                
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Parse response
            verdict = "NEUTRAL"
            confidence = 0 # Default to 0
            explanation = response_text
            
            # Extract verdict - handle bolding or other artifacts
            verdict_match = re.search(r'VERDICT:\s*\*?\*?\s*(SUPPORTS|REFUTES|NEUTRAL)', response_text, re.IGNORECASE)
            if verdict_match:
                verdict = verdict_match.group(1).upper()
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*\*?\*?\s*(\d+)', response_text)
            if confidence_match:
                confidence = max(0, min(100, int(confidence_match.group(1))))
            else:
                num_match = re.search(r'confidence.*?(\d+)', response_text, re.IGNORECASE | re.DOTALL)
                if num_match:
                    confidence = max(0, min(100, int(num_match.group(1))))
                else:
                    if verdict == "SUPPORTS": confidence = 75
                    elif verdict == "REFUTES": confidence = 85
                    else: confidence = 40
            
            # Extract explanation
            explanation_match = re.search(r'EXPLANATION:\s*\*?\*?\s*(.+)', response_text, re.DOTALL | re.IGNORECASE)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                # Concise 3-4 sentences
                lines = explanation.split('.')
                if len(lines) > 5:
                    explanation = '.'.join(lines[:4]).strip() + '.'
            
            if internal_verdict == "SUPPORTS" and verdict == "SUPPORTS":
                confidence = min(98, confidence + 5)
            
            if not confidence_match:
                print(f"  Warning: Could not parse confidence from SambaNova response. Defaulted to {confidence}.")
            
            return {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "explanation": explanation
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error verifying claim with SambaNova (attempt {attempt + 1}): {e}. Retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            
            print(f"Error verifying claim with SambaNova after {max_retries} attempts: {e}")
            return {
                "claim": claim,
                "verdict": "NEUTRAL",
                "confidence": 0,
                "explanation": f"Error during external fact-checking (SambaNova): {str(e)}"
            }

    return {
        "claim": claim,
        "verdict": "NEUTRAL",
        "confidence": 0,
        "explanation": "External fact-checking (SambaNova) failed after multiple attempts."
    }



def verify_claim_external(claim: str, internal_verdict: str = "NEUTRAL") -> Dict:
    """
    Verify claim using available external API (Bytez preferred, SambaNova fallback).
    """
    if USE_BYTEZ:
        return verify_claim_with_bytez(claim, internal_verdict)
    elif USE_SAMBANOVA:
        return verify_claim_with_sambanova(claim, internal_verdict)
    else:
        return {
            "claim": claim,
            "verdict": "NEUTRAL",
            "confidence": 0,
            "explanation": "No external APIs (Bytez or SambaNova) configured for fact-checking."
        }



def fact_check_qa_pair(qa_pair: Dict, transcript: str) -> Dict:
    """
    Fact-check all claims in a Q&A pair using two-layer verification.
    
    Args:
        qa_pair: Q&A pair dictionary with "answers" field
        transcript: Full transcript text for internal verification
        
    Returns:
        Q&A pair with "claims" field populated with verified claims
    """
    answer = qa_pair.get("answers", {}).get("long", "")
    
    if not answer or len(answer) < 30:
        qa_pair["claims"] = []
        return qa_pair
    
    claims_text = extract_claims_from_answer(answer)
    
    if not claims_text:
        qa_pair["claims"] = []
        return qa_pair
    
    print(f"Extracted {len(claims_text)} claims from Q&A pair for verification")
    
    # Verify each claim with two-layer approach
    verified_claims = []
    import time
    for i, claim_text in enumerate(claims_text):
        print(f"Verifying claim {i+1}/{len(claims_text)}: {claim_text[:60]}...")
        
        # Layer 1: Internal verification
        internal_verdict, internal_confidence = verify_claim_internal(claim_text, transcript)
        print(f"  Internal check: {internal_verdict} (confidence: {internal_confidence:.1f}%)")
        
        # Layer 2: External verification
        external_result = verify_claim_external(claim_text, internal_verdict)
        
        verified_claim = {
            "claim": claim_text,
            "verdict": external_result["verdict"],
            "confidence": external_result["confidence"],
            "explanation": external_result["explanation"],
            "internal_check": internal_verdict,
            "internal_confidence": internal_confidence
        }
        
        verified_claims.append(verified_claim)
        print(f"  External check: {external_result['verdict']} (confidence: {external_result['confidence']}%)")
        
        # Add small delay to prevent rate limiting (429)
        if i < len(claims_text) - 1:
            time.sleep(2)

    
    qa_pair["claims"] = verified_claims
    return qa_pair


def fact_check_qa_pairs(qa_pairs: List[Dict], transcript: str) -> List[Dict]:
    """
    Fact-check all Q&A pairs using two-layer verification.
    
    Args:
        qa_pairs: List of Q&A pair dictionaries
        transcript: Full transcript text for internal verification
        
    Returns:
        List of Q&A pairs with claims verified
    """
    providers = []
    if USE_BYTEZ: providers.append("Bytez")
    if USE_SAMBANOVA: providers.append("SambaNova")
    
    provider_str = " + ".join(providers) if providers else "None"
    print(f"Starting fact-checking for {len(qa_pairs)} Q&A pairs...")
    print(f"Using two-layer verification: Internal (transcript) + External ({provider_str} API)")
    
    for i, qa_pair in enumerate(qa_pairs):
        print(f"\nFact-checking Q&A pair {i+1}/{len(qa_pairs)}: {qa_pair.get('id', 'unknown')}")
        qa_pair = fact_check_qa_pair(qa_pair, transcript)
    
    print(f"\nFact-checking complete for {len(qa_pairs)} Q&A pairs")
    return qa_pairs

