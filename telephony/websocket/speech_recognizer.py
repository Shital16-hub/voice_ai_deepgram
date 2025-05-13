"""
Speech recognition handling for WebSocket connections using SimpleGoogleSTT.
"""
import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Callable, Awaitable
import numpy as np

from speech_to_text.simple_google_stt import SimpleGoogleSTT

logger = logging.getLogger(__name__)

# Enhanced patterns for non-speech annotations
NON_SPEECH_PATTERNS = [
    r'\(.*?music.*?\)',         # (music), (tense music), etc.
    r'\(.*?wind.*?\)',          # (wind), (wind blowing), etc.
    r'\(.*?engine.*?\)',        # (engine), (engine revving), etc.
    r'\(.*?noise.*?\)',         # (noise), (background noise), etc.
    r'\(.*?sound.*?\)',         # (sound), (sounds), etc.
    r'\(.*?silence.*?\)',       # (silence), etc.
    r'\[.*?silence.*?\]',       # [silence], etc.
    r'\[.*?BLANK.*?\]',         # [BLANK_AUDIO], etc.
    r'\(.*?applause.*?\)',      # (applause), etc.
    r'\(.*?laughter.*?\)',      # (laughter), etc.
    r'\(.*?footsteps.*?\)',     # (footsteps), etc.
    r'\(.*?breathing.*?\)',     # (breathing), etc.
    r'\(.*?growling.*?\)',      # (growling), etc.
    r'\(.*?coughing.*?\)',      # (coughing), etc.
    r'\(.*?clap.*?\)',          # (clap), etc.
    r'\(.*?laugh.*?\)',         # (laughing), etc.
    r'\[.*?noise.*?\]',         # [noise], etc.
    r'\(.*?background.*?\)',    # (background), etc.
    r'\[.*?music.*?\]',         # [music], etc.
    r'\(.*?static.*?\)',        # (static), etc.
    r'\[.*?unclear.*?\]',       # [unclear], etc.
    r'\(.*?inaudible.*?\)',     # (inaudible), etc.
    r'\<.*?noise.*?\>',         # <noise>, etc.
    r'music playing',           # Common transcription
    r'background noise',        # Common transcription
    r'static',                  # Common transcription
]

class SpeechRecognitionHandler:
    """Handles speech recognition for WebSocket connections using SimpleGoogleSTT."""
    
    def __init__(self):
        """Initialize the speech recognition handler."""
        self.speech_client = SimpleGoogleSTT(
            language_code="en-US",
            sample_rate=16000,
            enable_automatic_punctuation=True
        )
        
        # Compile the non-speech patterns for efficient use
        self.non_speech_pattern = re.compile('|'.join(NON_SPEECH_PATTERNS))
        
        # Speech recognition state
        self.is_active = False
        self.min_words_for_valid_query = 1
        
        # Echo detection
        self.recent_system_responses = []
        
        # Audio buffer for accumulating chunks
        self.audio_buffer = bytearray()
        self.buffer_lock = asyncio.Lock()
        self.min_buffer_size = 8000  # 0.5 seconds at 16kHz
        
        logger.info("Initialized SpeechRecognitionHandler with SimpleGoogleSTT")
    
    async def start_speech_session(self):
        """Start a new speech recognition session."""
        if not self.is_active:
            await self.speech_client.start_streaming()
            self.is_active = True
            logger.info("Started SimpleGoogleSTT streaming session")
    
    async def stop_speech_session(self):
        """Stop the speech recognition session."""
        if self.is_active:
            try:
                await self.speech_client.stop_streaming()
                self.is_active = False
                logger.info("Stopped SimpleGoogleSTT streaming session")
            except Exception as e:
                logger.error(f"Error stopping SimpleGoogleSTT streaming session: {e}")
    
    async def process_audio_chunk(
        self,
        audio_bytes: bytes,
        callback: Optional[Callable] = None
    ) -> List[Dict]:
        """Process audio chunk through speech recognition."""
        results = []
        
        # Add to buffer
        async with self.buffer_lock:
            self.audio_buffer.extend(audio_bytes)
            
            # Only process when we have enough audio
            if len(self.audio_buffer) < self.min_buffer_size:
                return results
            
            # Get accumulated audio
            audio_to_process = bytes(self.audio_buffer)
            self.audio_buffer.clear()
        
        try:
            # Ensure session is active
            if not self.is_active:
                await self.start_speech_session()
            
            # Process chunk using SimpleGoogleSTT's synchronous approach
            result = await self.speech_client.process_audio_chunk(
                audio_to_process,
                callback=callback
            )
            
            if result:
                # Create a result object for consistency
                result_obj = {
                    'text': result.text,
                    'is_final': True,  # SimpleGoogleSTT always returns final results
                    'confidence': getattr(result, 'confidence', 0.8)
                }
                results.append(result_obj)
                logger.debug(f"Received Google Speech result: {result.text}")
            
            return results
        except Exception as e:
            logger.error(f"Error during speech recognition: {e}")
            # Reset session on error
            await self._reset_speech_session()
            return []
    
    async def get_final_transcription(self) -> str:
        """Get final transcription results - not needed for SimpleGoogleSTT."""
        # SimpleGoogleSTT always returns final results
        return ""
    
    async def _reset_speech_session(self):
        """Reset the speech recognition session after an error."""
        if self.is_active:
            try:
                logger.info("Resetting SimpleGoogleSTT session after error")
                await self.speech_client.stop_streaming()
                await self.speech_client.start_streaming()
                self.is_active = True
            except Exception as session_error:
                logger.error(f"Error resetting SimpleGoogleSTT session: {session_error}")
                self.is_active = False
    
    def cleanup_transcription(self, text: str) -> str:
        """Clean up transcription text."""
        if not text:
            return ""
        
        # Remove non-speech annotations
        cleaned_text = self.non_speech_pattern.sub('', text)
        
        # Remove common filler words at beginning
        cleaned_text = re.sub(r'^(um|uh|er|ah|like|so)\s+', '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove repeated words (stuttering)
        cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned_text)
        
        # Clean up punctuation
        cleaned_text = re.sub(r'\s+([.,!?])', r'\1', cleaned_text)
        
        # Clean up double spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Log what was cleaned if substantial
        if text != cleaned_text:
            logger.info(f"Cleaned transcription: '{text}' -> '{cleaned_text}'")
        
        return cleaned_text
    
    def is_valid_transcription(self, text: str) -> bool:
        """Check if a transcription is valid."""
        # Clean up the text first
        cleaned_text = self.cleanup_transcription(text)
        
        # Check if empty after cleaning
        if not cleaned_text:
            logger.debug("Transcription contains only non-speech annotations")
            return False
        
        # Check word count
        word_count = len(cleaned_text.split())
        if word_count < self.min_words_for_valid_query:
            logger.debug(f"Transcription too short: {word_count} words")
            return False
        
        return True
    
    def is_echo_of_system_speech(self, transcription: str) -> bool:
        """Check if transcription is an echo of system speech."""
        if not transcription:
            return False
        
        # Check recent responses for similarity
        for phrase in self.recent_system_responses:
            clean_phrase = self.cleanup_transcription(phrase)
            
            if clean_phrase and len(clean_phrase) > 5:
                # Check for substring match
                if clean_phrase in transcription or transcription in clean_phrase:
                    similarity_ratio = len(clean_phrase) / max(len(transcription), 1)
                    
                    if similarity_ratio > 0.5:  # At least 50% match
                        logger.info(f"Detected echo of system speech: '{clean_phrase}' similar to '{transcription}'")
                        return True
        
        return False
    
    def add_system_response(self, response: str):
        """Add a system response for echo detection."""
        self.recent_system_responses.append(response)
        if len(self.recent_system_responses) > 5:  # Keep last 5 responses
            self.recent_system_responses.pop(0)