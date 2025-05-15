"""
Speech processor using Google Cloud STT v2 with proper async handling.
"""
import logging
import asyncio
import time
import os
import json
from typing import Dict, Any, Optional, List, Callable, Awaitable

# Import the updated Google Cloud STT
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """
    Speech processor optimized for telephony using Google Cloud STT v2.
    Zero preprocessing - trust the telephony-optimized API.
    """
    
    def __init__(self, pipeline):
        """Initialize speech processor."""
        self.pipeline = pipeline
        
        logger.info("Initializing SpeechProcessor with v2 API")
        
        # Get project ID
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        if not project_id:
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_file and os.path.exists(credentials_file):
                try:
                    with open(credentials_file, 'r') as f:
                        creds_data = json.load(f)
                        project_id = creds_data.get('project_id')
                        logger.info(f"Extracted project ID: {project_id}")
                except Exception as e:
                    logger.error(f"Error reading credentials: {e}")
        
        if not project_id:
            raise ValueError("Google Cloud project ID is required")
        
        # Initialize Speech STT v2 with telephony optimization
        self.speech_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,  # Match Twilio
            encoding="MULAW",   # Match Twilio
            channels=1,
            interim_results=False,  # Only final results
            project_id=project_id,
            enhanced_model=True,    # Use telephony model
            location="global"
        )
        
        # Echo detection
        self.echo_history = []
        self.max_echo_history = 3
        
        # Statistics
        self.audio_chunks_received = 0
        self.successful_transcriptions = 0
        self.failed_transcriptions = 0
        
        logger.info("SpeechProcessor initialized with Google Cloud v2 telephony")
    
    async def process_audio(
        self,
        audio_data: bytes,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Optional[str]:
        """
        Process audio with zero modification - let Google handle everything.
        """
        self.audio_chunks_received += 1
        
        try:
            logger.debug(f"Processing audio chunk #{self.audio_chunks_received}, "
                        f"size: {len(audio_data)} bytes")
            
            # Collect results
            final_result = None
            
            async def collect_result(result):
                nonlocal final_result
                if result.is_final:
                    final_result = result
                if callback:
                    await callback(result)
            
            # Pass audio directly to Google Cloud STT v2
            result = await self.speech_client.process_audio_chunk(
                audio_chunk=audio_data,
                callback=collect_result
            )
            
            # Check for final result
            if final_result and final_result.text:
                self.successful_transcriptions += 1
                cleaned_text = final_result.text.strip()
                
                if cleaned_text:
                    logger.info(f"Transcription: '{cleaned_text}' (confidence: {final_result.confidence:.2f})")
                    return cleaned_text
            else:
                self.failed_transcriptions += 1
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            self.failed_transcriptions += 1
            return None
    
    def cleanup_transcription(self, text: str) -> str:
        """Minimal cleanup - trust Google's telephony model."""
        if not text:
            return ""
        
        # Only basic normalization
        cleaned = text.strip()
        
        # Ensure first letter is capitalized
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def is_valid_transcription(self, text: str) -> bool:
        """Validate transcription with minimal requirements."""
        cleaned = text.strip() if text else ""
        
        if not cleaned:
            return False
        
        # Must have at least one character
        if len(cleaned) < 1:
            return False
        
        # Must have at least one letter
        if not any(c.isalpha() for c in cleaned):
            return False
        
        return True
    
    def is_echo_of_system_speech(self, transcription: str) -> bool:
        """Simple echo detection."""
        if not transcription or not self.echo_history:
            return False
        
        normalized_trans = transcription.lower().strip()
        
        for recent_response in self.echo_history:
            normalized_response = recent_response.lower().strip()
            
            # Exact match
            if normalized_trans == normalized_response:
                logger.info(f"Echo detected: '{transcription}'")
                return True
            
            # Word overlap check
            trans_words = set(normalized_trans.split())
            response_words = set(normalized_response.split())
            
            if trans_words and response_words:
                overlap_ratio = len(trans_words & response_words) / len(trans_words)
                if overlap_ratio > 0.8:
                    logger.info(f"Echo detected (overlap): '{transcription}'")
                    return True
        
        return False
    
    def add_to_echo_history(self, response: str) -> None:
        """Add response to echo history."""
        if response and len(response) > 5:
            self.echo_history.append(response)
            if len(self.echo_history) > self.max_echo_history:
                self.echo_history.pop(0)
            logger.debug(f"Added to echo history: '{response[:30]}...'")
    
    async def stop_speech_session(self) -> None:
        """Stop the speech session."""
        await self.speech_client.stop_streaming()
        
        # Log statistics
        success_rate = (self.successful_transcriptions / max(self.audio_chunks_received, 1)) * 100
        logger.info(f"Speech session ended - Success rate: {success_rate:.1f}%")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "audio_chunks_received": self.audio_chunks_received,
            "successful_transcriptions": self.successful_transcriptions,
            "failed_transcriptions": self.failed_transcriptions,
            "echo_history_size": len(self.echo_history),
            "success_rate": round((self.successful_transcriptions / max(self.audio_chunks_received, 1)) * 100, 2),
            "stt_stats": self.speech_client.get_stats()
        }