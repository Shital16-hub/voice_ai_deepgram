"""
Echo detection for system speech.
"""
import logging
from typing import List

logger = logging.getLogger(__name__)

class EchoDetection:
    """Detects when transcribed text is an echo of the system's own speech."""
    
    def __init__(self, max_recent_responses=5):
        self.recent_system_responses = []
        self.max_recent_responses = max_recent_responses
    
    def add_system_response(self, response: str) -> None:
        """
        Add a system response to track for echo detection.
        
        Args:
            response: System response text
        """
        # Clean the response for comparison
        from telephony.speech.transcription_cleaner import TranscriptionCleaner
        cleaner = TranscriptionCleaner()
        cleaned_response = cleaner.cleanup_transcription(response)
        
        # Add to recent responses
        self.recent_system_responses.append(cleaned_response)
        
        # Keep only recent responses
        if len(self.recent_system_responses) > self.max_recent_responses:
            self.recent_system_responses.pop(0)
    
    def is_echo(self, transcription: str) -> bool:
        """
        Check if a transcription appears to be an echo of the system's own speech.
        
        Args:
            transcription: The transcription to check
            
        Returns:
            True if the transcription appears to be an echo
        """
        # No transcription, no echo
        if not transcription:
            return False
        
        # Check recent responses for similarity
        for phrase in self.recent_system_responses:
            # Clean up response text for comparison
            clean_phrase = phrase.strip()
            
            # Check for substring match
            if clean_phrase and len(clean_phrase) > 5:
                # If transcription contains a significant part of our recent speech
                if clean_phrase in transcription or transcription in clean_phrase:
                    similarity_ratio = len(clean_phrase) / max(len(transcription), 1)
                    
                    if similarity_ratio > 0.5:  # At least 50% match
                        logger.info(f"Detected echo of system speech: '{clean_phrase}' similar to '{transcription}'")
                        return True
                    
        return False
    
    def reset(self) -> None:
        """Reset echo detection state."""
        self.recent_system_responses.clear()