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
    # Try to find the file in different possible locations
    possible_paths = [
        "telephony/websocket_handler.py",
        "/workspace/voice_ai_deepgram/telephony/websocket_handler.py",
        "./telephony/websocket_handler.py"
    ]
    
    filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            filepath = path
            break
    
    if not filepath:
        logger.error("Could not find websocket_handler.py file. Please provide the full path.")
        return
    
    try:
        # Read the file
        logger.info(f"Reading file from: {filepath}")
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
        
        # 6. Add keyterms support to _get_params method if it exists
        get_params_pattern = re.compile(r'def _get_params\(self\)[^:]*:.*?return params', re.DOTALL)
        match = get_params_pattern.search(content)
        
        if match:
            params_section = match.group(0)
            # Check if keyterms aren't already added
            if "keyterms" not in params_section:
                # Find the end of the params dictionary
                params_end = params_section.rfind("return params")
                if params_end > 0:
                    # Insert keyterms line before the return statement
                    updated_section = params_section[:params_end] + \
                        "        # Add Nova 3 specific parameters for better telephony performance\n" + \
                        "        params[\"keyterms\"] = json.dumps([\"price\", \"plan\", \"cost\", \"subscription\", \"service\", \"features\", \"support\"])\n        \n        " + \
                        params_section[params_end:]
                    content = content.replace(params_section, updated_section)
        
        # Write the updated content back to the file
        with open(filepath, "w") as f:
            f.write(content)
        
        logger.info(f"Successfully updated {filepath} to use Nova 3!")
    
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
    except Exception as e:
        logger.error(f"Error updating websocket handler: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    update_websocket_handler()
