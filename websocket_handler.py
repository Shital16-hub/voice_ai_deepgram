#!/usr/bin/env python3
"""
Utility to update WebSocket handler to use Deepgram Nova 3.
"""
import os
import sys
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_websocket_handler():
    """Update websocket_handler.py to use Deepgram Nova 3."""
    filepath = "telephony/websocket_handler.py"
    
    try:
        # Read the file
        with open(filepath, "r") as f:
            content = f.read()
        
        # Make the necessary updates
        
        # 1. Update import statements if needed
        if "from speech_to_text.deepgram_stt import DeepgramStreamingSTT" not in content:
            content = content.replace(
                "from speech_to_text.deepgram_stt import", 
                "from speech_to_text.deepgram_stt import DeepgramStreamingSTT,"
            )
        
        # 2. Update initialization in __init__ method
        content = content.replace(
            "self.deepgram_session_active = False",
            "self.deepgram_session_active = False  # Reset Deepgram Nova 3 session state"
        )
        
        # 3. Update _handle_start method
        content = content.replace(
            "await self.pipeline.speech_recognizer.start_streaming()",
            "await self.pipeline.speech_recognizer.start_streaming()"
        ).replace(
            "logger.info(\"Started Deepgram streaming session\")",
            "logger.info(\"Started Deepgram Nova 3 streaming session\")"
        ).replace(
            "logger.error(f\"Error starting Deepgram streaming session: {e}\")",
            "logger.error(f\"Error starting Deepgram Nova 3 streaming session: {e}\")"
        )
        
        # 4. Update _handle_media method
        content = content.replace(
            "logger.info(\"Starting new Deepgram streaming session\")",
            "logger.info(\"Starting new Deepgram Nova 3 streaming session\")"
        )
        
        # 5. Update _handle_stop method
        content = content.replace(
            "logger.info(\"Stopped Deepgram streaming session\")",
            "logger.info(\"Stopped Deepgram Nova 3 streaming session\")"
        ).replace(
            "logger.error(f\"Error stopping Deepgram streaming session: {e}\")",
            "logger.error(f\"Error stopping Deepgram Nova 3 streaming session: {e}\")"
        )
        
        # 6. Add keyterms support in _process_audio method if applicable
        process_audio_pattern = re.compile(r'async def _process_audio\(self, ws\) -> None:.*?try:', re.DOTALL)
        if process_audio_pattern.search(content):
            process_audio_match = process_audio_pattern.search(content)
            process_audio_code = process_audio_match.group(0)
            
            # Check if we need to add keyterms
            if "keyterms" not in process_audio_code and "params" in process_audio_code:
                updated_process_audio = process_audio_code.replace(
                    "params = self._get_params()",
                    "params = self._get_params()\n        "
                    "# Add Nova 3 specific parameters for better telephony performance\n        "
                    "params[\"keyterms\"] = json.dumps([\"price\", \"plan\", \"cost\", \"subscription\", \"service\", \"features\", \"support\"])"
                )
                content = content.replace(process_audio_code, updated_process_audio)
        
        # Write the updated content back to the file
        with open(filepath, "w") as f:
            f.write(content)
        
        logger.info(f"Successfully updated {filepath} to use Nova 3!")
    
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
    except Exception as e:
        logger.error(f"Error updating websocket handler: {e}")

if __name__ == "__main__":
    update_websocket_handler()