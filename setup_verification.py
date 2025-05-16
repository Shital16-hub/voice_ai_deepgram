#!/usr/bin/env python3
"""
Setup script to initialize and test the Voice AI Agent system.
Ensures all components are properly configured and working.
"""
import os
import sys
import asyncio
import logging
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def verify_google_cloud_setup():
    """Verify Google Cloud credentials and project setup."""
    logger.info("Verifying Google Cloud setup...")
    
    # Check credentials file
    credentials_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_file:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        return False
    
    if not os.path.exists(credentials_file):
        logger.error(f"Credentials file not found: {credentials_file}")
        return False
    
    # Load and verify credentials
    try:
        with open(credentials_file, 'r') as f:
            creds_data = json.load(f)
            project_id = creds_data.get('project_id')
            if not project_id:
                logger.error("project_id not found in credentials file")
                return False
            
            logger.info(f"Found project ID: {project_id}")
            
            # Set environment variable if not already set
            if not os.getenv('GOOGLE_CLOUD_PROJECT'):
                os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
                logger.info(f"Set GOOGLE_CLOUD_PROJECT to {project_id}")
    
    except Exception as e:
        logger.error(f"Error reading credentials file: {e}")
        return False
    
    # Test Google Cloud Speech client
    try:
        from google.cloud.speech_v2 import SpeechClient
        client = SpeechClient()
        logger.info("✓ Google Cloud Speech v2 client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Speech client: {e}")
        return False
    
    # Test Google Cloud TTS client
    try:
        from google.cloud import texttospeech
        tts_client = texttospeech.TextToSpeechClient()
        logger.info("✓ Google Cloud TTS client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing TTS client: {e}")
        return False
    
    return True

async def test_stt_component():
    """Test the STT component with the fixed implementation."""
    logger.info("Testing STT component...")
    
    try:
        # Import the fixed STT component
        from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT
        from speech_to_text.stt_integration import STTIntegration
        
        # Test project ID extraction
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if not project_id:
            credentials_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if credentials_file and os.path.exists(credentials_file):
                with open(credentials_file, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
        
        if not project_id:
            logger.error("Could not determine project ID")
            return False
        
        # Initialize STT with telephony settings
        stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,
            project_id=project_id,
            location="global"
        )
        
        # Test STT integration
        stt_integration = STTIntegration(
            speech_recognizer=stt_client,
            language="en-US"
        )
        await stt_integration.init(project_id=project_id)
        
        logger.info("✓ STT component initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing STT component: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_tts_component():
    """Test the TTS component with the fixed implementation."""
    logger.info("Testing TTS component...")
    
    try:
        from text_to_speech.google_cloud_tts import GoogleCloudTTS
        
        # Initialize TTS with Twilio-optimized settings
        credentials_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        tts_client = GoogleCloudTTS(
            credentials_file=credentials_file,
            voice_name="en-US-Neural2-C",
            voice_gender=None,  # Don't set for Neural2 voices
            language_code="en-US",
            container_format="mulaw",
            sample_rate=8000,
            enable_caching=True,
            voice_type="NEURAL2"
        )
        
        # Test synthesis with a simple phrase
        test_text = "Hello, this is a test of the text to speech system."
        audio_data = await tts_client.synthesize(test_text)
        
        if audio_data and len(audio_data) > 0:
            logger.info(f"✓ TTS component working - Generated {len(audio_data)} bytes")
            return True
        else:
            logger.error("TTS generated empty audio")
            return False
        
    except Exception as e:
        logger.error(f"Error testing TTS component: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_knowledge_base():
    """Test the knowledge base component."""
    logger.info("Testing knowledge base component...")
    
    try:
        from knowledge_base.llama_index.document_store import DocumentStore
        from knowledge_base.llama_index.index_manager import IndexManager
        from knowledge_base.llama_index.query_engine import QueryEngine
        from knowledge_base.conversation_manager import ConversationManager
        
        # Initialize document store
        doc_store = DocumentStore()
        
        # Initialize index manager with storage
        storage_dir = './storage'
        os.makedirs(storage_dir, exist_ok=True)
        index_manager = IndexManager(storage_dir=storage_dir)
        await index_manager.init()
        
        # Initialize query engine
        query_engine = QueryEngine(
            index_manager=index_manager, 
            llm_model_name='mistral:7b-instruct-v0.2-q4_0',
            llm_temperature=0.7
        )
        await query_engine.init()
        
        # Initialize conversation manager
        conversation_manager = ConversationManager(
            query_engine=query_engine,
            llm_model_name='mistral:7b-instruct-v0.2-q4_0',
            llm_temperature=0.7,
            skip_greeting=True
        )
        await conversation_manager.init()
        
        logger.info("✓ Knowledge base components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing knowledge base: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def initialize_voice_ai_agent():
    """Initialize the complete Voice AI Agent system."""
    logger.info("Initializing Voice AI Agent system...")
    
    try:
        from voice_ai_agent import VoiceAIAgent
        
        # Initialize agent with optimized settings
        agent = VoiceAIAgent(
            storage_dir='./storage',
            model_name='mistral:7b-instruct-v0.2-q4_0',
            llm_temperature=0.7,
            tts_voice_name="en-US-Neural2-C",
            tts_language_code="en-US"
        )
        
        await agent.init()
        
        if agent.initialized:
            logger.info("✓ Voice AI Agent initialized successfully")
            return True
        else:
            logger.error("Voice AI Agent initialization failed")
            return False
        
    except Exception as e:
        logger.error(f"Error initializing Voice AI Agent: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Main setup and verification process."""
    logger.info("Starting Voice AI Agent setup and verification...")
    
    # List of verification steps
    verification_steps = [
        ("Google Cloud Setup", verify_google_cloud_setup),
        ("STT Component", test_stt_component),
        ("TTS Component", test_tts_component),
        ("Knowledge Base", test_knowledge_base),
        ("Voice AI Agent", initialize_voice_ai_agent),
    ]
    
    results = {}
    for step_name, step_func in verification_steps:
        logger.info(f"\n{'='*50}")
        logger.info(f"Verifying: {step_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await step_func()
            results[step_name] = result
            
            if result:
                logger.info(f"✓ {step_name}: PASSED")
            else:
                logger.error(f"✗ {step_name}: FAILED")
                
        except Exception as e:
            logger.error(f"✗ {step_name}: ERROR - {e}")
            results[step_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("VERIFICATION SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for step_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{step_name:.<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} components verified successfully")
    
    if passed == total:
        logger.info("\n🎉 All components verified successfully!")
        logger.info("The Voice AI Agent system is ready for use with Twilio.")
    else:
        logger.error(f"\n❌ {total - passed} component(s) failed verification.")
        logger.error("Please check the errors above and fix the issues.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)