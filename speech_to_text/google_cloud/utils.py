"""
Utility functions for Google Cloud Speech-to-Text integration.
"""
import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)

def format_recognition_config(
    sample_rate: int,
    language_code: str,
    model: str = "latest_long",
    encoding: str = "LINEAR16",
    **kwargs
) -> Dict[str, Any]:
    """
    Format a recognition config for Google Cloud Speech-to-Text.
    
    Args:
        sample_rate: Audio sample rate in Hz
        language_code: Language code (e.g., 'en-US')
        model: Model name
        encoding: Audio encoding
        **kwargs: Additional parameters
        
    Returns:
        Properly formatted recognition config
    """
    config = {
        "sample_rate_hertz": sample_rate,
        "language_code": language_code,
        "model": model,
        "encoding": encoding.upper()
    }
    
    # Add optional parameters
    for key, value in kwargs.items():
        if value is not None:
            # Convert snake_case to camelCase for API
            api_key = ''.join(word.capitalize() if i > 0 else word 
                            for i, word in enumerate(key.split('_')))
            config[api_key] = value
    
    return config

def parse_streaming_response(response) -> Dict[str, Any]:
    """
    Parse a Google Cloud Speech-to-Text streaming response.
    
    Args:
        response: Response from the API
        
    Returns:
        Parsed response as a dictionary
    """
    result = {
        "results": [],
        "is_final": False,
        "text": "",
        "confidence": 0.0
    }
    
    try:
        # Extract results
        for res in response.results:
            # Create a result entry
            result_entry = {
                "is_final": res.is_final,
                "alternatives": []
            }
            
            # Extract alternatives
            for alt in res.alternatives:
                alt_entry = {
                    "transcript": alt.transcript,
                    "confidence": alt.confidence
                }
                
                # Add words if available
                if hasattr(alt, 'words') and alt.words:
                    alt_entry["words"] = [
                        {
                            "word": word_info.word,
                            "start_time": word_info.start_time.total_seconds(),
                            "end_time": word_info.end_time.total_seconds()
                        }
                        for word_info in alt.words
                    ]
                
                result_entry["alternatives"].append(alt_entry)
            
            result["results"].append(result_entry)
            
            # Update top-level fields for convenience
            if res.is_final and res.alternatives:
                result["is_final"] = True
                result["text"] = res.alternatives[0].transcript
                result["confidence"] = res.alternatives[0].confidence
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing streaming response: {e}")
        return result

def create_speech_context(
    phrases: List[str],
    boost: float = 15.0
) -> Dict[str, Any]:
    """
    Create a speech context for better recognition.
    
    Args:
        phrases: List of phrases to boost
        boost: Boost factor (0-20)
        
    Returns:
        Formatted speech context
    """
    return {
        "phrases": phrases,
        "boost": min(20.0, max(0.0, boost))  # Clamp to 0-20 range
    }

def save_credentials_from_json(json_str: str, output_path: str) -> bool:
    """
    Save Google Cloud credentials from a JSON string to a file.
    
    Args:
        json_str: JSON credentials string
        output_path: Path to save the credentials file
        
    Returns:
        True if successful
    """
    try:
        # Validate JSON
        credentials = json.loads(json_str)
        
        # Check for required fields
        required_fields = ["type", "project_id", "private_key_id", "private_key", 
                          "client_email", "client_id"]
        
        for field in required_fields:
            if field not in credentials:
                logger.error(f"Missing required field in credentials: {field}")
                return False
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        logger.info(f"Saved Google Cloud credentials to {output_path}")
        return True
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON credentials string")
        return False
    except Exception as e:
        logger.error(f"Error saving credentials: {e}")
        return False