#!/usr/bin/env python3
import asyncio
import os
import aiohttp
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_available_models():
    """Test available Deepgram models with current API key"""
    
    # Get API key from environment
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("DEEPGRAM_API_KEY not set in environment")
        return
    
    logger.info(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
    logger.info("Testing Deepgram API connection...")
    
    headers = {
        "Authorization": f"Token {api_key}"
    }
    
    # Request to get projects (to verify API key works)
    url = "https://api.deepgram.com/v1/projects"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    projects_data = await response.json()
                    logger.info("API connection successful!")
                    logger.info(f"Projects data: {projects_data}")
                    
                    # Test a simple transcription to check what models are accessible
                    logger.info("\nTesting a simple transcription with 'general' model...")
                    
                    # Create a simple audio file with text content for testing
                    test_text = "This is a test of the Deepgram API."
                    
                    # Test with general model
                    transcription_url = "https://api.deepgram.com/v1/listen?model=general"
                    test_data = {
                        "text": test_text
                    }
                    
                    try:
                        async with session.post(
                            transcription_url, 
                            headers=headers, 
                            json=test_data
                        ) as trans_response:
                            logger.info(f"Response status: {trans_response.status}")
                            if trans_response.status == 200:
                                logger.info("General model is available!")
                            else:
                                resp_text = await trans_response.text()
                                logger.info(f"General model test response: {resp_text}")
                    except Exception as e:
                        logger.error(f"Error testing general model: {e}")
                    
                    # For now, recommend using the 'general' model
                    logger.info("\nRecommended model for your project: 'general'")
                    logger.info("To use this model, update your .env file:")
                    logger.info("STT_MODEL_NAME=general")
                else:
                    error = await response.text()
                    logger.error(f"API error: {response.status} - {error}")
    
    except Exception as e:
        logger.error(f"Error connecting to Deepgram API: {e}")

if __name__ == "__main__":
    asyncio.run(test_available_models())