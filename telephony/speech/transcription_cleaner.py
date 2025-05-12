"""
Transcription cleaning utilities.
"""
import re
import logging

logger = logging.getLogger(__name__)

class TranscriptionCleaner:
    """Cleans up transcription text by removing noise and non-speech annotations."""
    
    def __init__(self):
        # Enhanced patterns for non-speech annotations
        self.non_speech_patterns = [
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
        
        # Compile patterns for efficiency
        self.non_speech_pattern = re.compile('|'.join(self.non_speech_patterns))
    
    def cleanup_transcription(self, text: str) -> str:
        """
        Enhanced cleanup of transcription text by removing non-speech annotations and filler words.
        
        Args:
            text: Original transcription text
            
        Returns:
            Cleaned transcription text
        """
        if not text:
            return ""
            
        # Remove non-speech annotations
        cleaned_text = self.non_speech_pattern.sub('', text)
        
        # Remove common filler words at beginning of sentences
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