"""
Data models for the Google Cloud Speech-to-Text integration.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class TranscriptionConfig(BaseModel):
    """Configuration for transcription requests."""
    
    model: Optional[str] = Field(
        default=None,
        description="Google Cloud model to use"
    )
    
    language_code: Optional[str] = Field(
        default=None,
        description="Language code for transcription"
    )
    
    alternative_language_codes: Optional[List[str]] = Field(
        default=None,
        description="Alternative language codes"
    )
    
    max_alternatives: Optional[int] = Field(
        default=None,
        description="Maximum number of alternative transcriptions"
    )
    
    profanity_filter: Optional[bool] = Field(
        default=None,
        description="Filter profanity from transcription"
    )
    
    enable_word_time_offsets: Optional[bool] = Field(
        default=None,
        description="Include word-level timestamps"
    )
    
    enable_automatic_punctuation: Optional[bool] = Field(
        default=None,
        description="Enable automatic punctuation"
    )
    
    enable_speaker_diarization: Optional[bool] = Field(
        default=None,
        description="Enable speaker diarization"
    )
    
    diarization_speaker_count: Optional[int] = Field(
        default=None,
        description="Number of speakers for diarization"
    )
    
    speech_contexts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Speech contexts for improved recognition"
    )
    
    use_enhanced: Optional[bool] = Field(
        default=None,
        description="Use enhanced model"
    )

class TranscriptionResult(BaseModel):
    """Result of a transcription request."""
    
    text: str = Field(
        description="Transcribed text"
    )
    
    confidence: float = Field(
        default=0.0,
        description="Confidence score (0.0-1.0)"
    )
    
    words: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Word-level details"
    )
    
    alternatives: List[str] = Field(
        default_factory=list,
        description="Alternative transcripts"
    )
    
    audio_duration: float = Field(
        default=0.0,
        description="Duration of the audio in seconds"
    )
    
    def get_words_with_times(self) -> List[Dict[str, Any]]:
        """Get words with their timestamps."""
        return [
            {
                "word": word.get("word", ""),
                "start": word.get("start_time", 0.0),
                "end": word.get("end_time", 0.0),
                "confidence": word.get("confidence", 0.0)
            }
            for word in self.words
        ]