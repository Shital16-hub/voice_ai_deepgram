#!/usr/bin/env python3
"""
Utility to update Deepgram STT configuration to Nova 3.
"""
import os
import sys
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_deepgram_config(model_name="nova-3", language="en-US"):
    """Update Deepgram configuration in config files."""
    # Update telephony/config.py
    try:
        with open("telephony/config.py", "r") as f:
            config_content = f.read()
        
        # Replace STT model setting - handle different possible formats
        if 'STT_MODEL = ' in config_content:
            # Replace if setting already exists
            config_content = config_content.replace(
                'STT_MODEL = "general"', 
                f'STT_MODEL = "{model_name}"'
            ).replace(
                "STT_MODEL = 'general'", 
                f"STT_MODEL = '{model_name}'"
            ).replace(
                'STT_MODEL = "nova-2"', 
                f'STT_MODEL = "{model_name}"'
            ).replace(
                "STT_MODEL = 'nova-2'", 
                f"STT_MODEL = '{model_name}'"
            )
        else:
            # Add setting if it doesn't exist
            new_setting = f'\n# STT Optimization Settings for Nova 3\nSTT_MODEL = "{model_name}"  # Use Nova 3 model'
            config_content += new_setting
        
        with open("telephony/config.py", "w") as f:
            f.write(config_content)
        
        logger.info(f"Updated STT model to {model_name} in telephony/config.py")
    except Exception as e:
        logger.error(f"Error updating telephony/config.py: {e}")
    
    # Update speech_to_text/deepgram_stt/streaming.py
    try:
        with open("speech_to_text/deepgram_stt/streaming.py", "r") as f:
            streaming_content = f.read()
        
        # Update the model_name default in __init__
        streaming_content = streaming_content.replace(
            'model_name: Optional[str] = None,',
            f'model_name: Optional[str] = "{model_name}",'
        ).replace(
            'model_name: Optional[str] = "general",',
            f'model_name: Optional[str] = "{model_name}",'
        ).replace(
            'model_name: Optional[str] = "nova-2",',
            f'model_name: Optional[str] = "{model_name}",'
        )
        
        with open("speech_to_text/deepgram_stt/streaming.py", "w") as f:
            f.write(streaming_content)
            
        logger.info(f"Updated default model to {model_name} in speech_to_text/deepgram_stt/streaming.py")
    except Exception as e:
        logger.error(f"Error updating streaming.py: {e}")
    
    # Update voice_ai_agent.py
    try:
        with open("voice_ai_agent.py", "r") as f:
            agent_content = f.read()
        
        # Update initialization for Nova 3
        if 'self.stt_model = ' in agent_content:
            agent_content = agent_content.replace(
                'self.stt_model = kwargs.get(\'stt_model\', \'general\')',
                f'self.stt_model = kwargs.get(\'stt_model\', \'{model_name}\')'
            ).replace(
                'self.stt_model = kwargs.get(\'stt_model\', "general")',
                f'self.stt_model = kwargs.get(\'stt_model\', "{model_name}")'
            ).replace(
                'self.stt_model = kwargs.get(\'stt_model\', \'nova-2\')',
                f'self.stt_model = kwargs.get(\'stt_model\', \'{model_name}\')'
            ).replace(
                'self.stt_model = kwargs.get(\'stt_model\', "nova-2")',
                f'self.stt_model = kwargs.get(\'stt_model\', "{model_name}")'
            )
        
        # Update logging messages to mention Nova 3
        agent_content = agent_content.replace(
            'Initializing Voice AI Agent components with Deepgram STT...',
            'Initializing Voice AI Agent components with Deepgram Nova 3 STT...'
        ).replace(
            'Voice AI Agent initialization complete with Deepgram STT',
            'Voice AI Agent initialization complete with Deepgram Nova 3 STT'
        )
        
        with open("voice_ai_agent.py", "w") as f:
            f.write(agent_content)
            
        logger.info(f"Updated Voice AI Agent to use {model_name}")
    except Exception as e:
        logger.error(f"Error updating voice_ai_agent.py: {e}")
    
    logger.info("Configuration update complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update Deepgram STT configuration to Nova 3")
    parser.add_argument("--model", default="nova-3", help="Deepgram model name")
    parser.add_argument("--language", default="en-US", help="Language code")
    
    args = parser.parse_args()
    update_deepgram_config(args.model, args.language)