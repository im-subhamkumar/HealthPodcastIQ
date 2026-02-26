"""Transcription service using OpenAI Whisper."""
import os
from pathlib import Path
from typing import Optional, Tuple
import re

# Try to import dependencies with helpful error messages
try:
    import whisper
except ImportError:
    raise ImportError(
        "Whisper module not found. Please install it with: pip install openai-whisper"
    )

try:
    import yt_dlp
except ImportError:
    raise ImportError(
        "yt-dlp module not found. Please install it with: pip install yt-dlp"
    )

from backend.config import TEMP_DIR, WHISPER_MODEL

# Global model instance (lazy loaded)
_whisper_model = None


def get_whisper_model():
    """Get or load Whisper model (singleton pattern)."""
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def download_youtube_audio(url: str, output_path: Path) -> Path:
    """
    Download audio from YouTube URL using yt-dlp.
    
    Args:
        url: YouTube URL
        output_path: Directory to save audio file
        
    Returns:
        Path to downloaded audio file
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # yt-dlp adds extension, but FFmpeg changes it to wav
        audio_path = Path(filename).with_suffix('.wav')
        return audio_path


def clean_transcript(text: str) -> str:
    """
    Clean transcript by removing filler words and fixing common issues.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned transcript
    """
    # Remove common filler words
    filler_words = ['um', 'uh', 'you know', 'like', 'so', 'well', 'actually', 'basically']
    words = text.split()
    cleaned_words = [w for w in words if w.lower() not in filler_words]
    
    # Fix common scientific misspellings (basic dictionary)
    scientific_fixes = {
        'mitochondria': 'mitochondria',
        'mitochondrial': 'mitochondrial',
        'adenosine': 'adenosine',
        'cortisol': 'cortisol',
        'testosterone': 'testosterone',
    }
    
    # Join and fix spacing
    cleaned_text = ' '.join(cleaned_words)
    
    # Remove excessive whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()


def transcribe_audio(audio_path: Path) -> str:
    """
    Transcribe audio file using Whisper.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Transcribed text
    """
    model = get_whisper_model()
    print(f"Transcribing audio: {audio_path}")
    
    result = model.transcribe(
        str(audio_path),
        language='en',
        task='transcribe',
        verbose=False
    )
    
    transcript = result['text']
    cleaned_transcript = clean_transcript(transcript)
    
    return cleaned_transcript


def process_source(source: str) -> Tuple[str, Optional[str]]:
    """
    Process input source (YouTube URL or file path) and return transcript.
    
    Args:
        source: YouTube URL or file path
        
    Returns:
        Tuple of (transcript, thumbnail_url)
    """
    thumbnail_url = None
    
    # Check if it's a YouTube URL
    youtube_pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.match(youtube_pattern, source)
    
    if match:
        video_id = match.group(1)
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        
        # Download audio
        print(f"Downloading YouTube video: {source}")
        audio_path = download_youtube_audio(source, TEMP_DIR)
    else:
        # Assume it's a file path
        audio_path = Path(source)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")
    
    # Transcribe
    transcript = transcribe_audio(audio_path)
    
    # Clean up temporary file if it was downloaded
    if match and audio_path.exists():
        try:
            audio_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete temp file {audio_path}: {e}")
    
    return transcript, thumbnail_url

