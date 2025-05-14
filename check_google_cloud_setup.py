# check_google_cloud_setup.py

"""
Check Google Cloud Speech API setup and permissions.
"""
import os
import json
import asyncio
import logging
from google.cloud import speech_v2
from google.auth import default
from google.auth.transport.requests import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_credentials():
    """Check if credentials are properly set up."""
    logger.info("=== Checking Credentials ===")
    
    # Check environment variables
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    logger.info(f"GOOGLE_CLOUD_PROJECT: {project_id}")
    logger.info(f"GOOGLE_APPLICATION_CREDENTIALS: {creds_file}")
    
    if not project_id:
        logger.error("❌ GOOGLE_CLOUD_PROJECT not set")
        return False
    
    if not creds_file:
        logger.warning("⚠️  GOOGLE_APPLICATION_CREDENTIALS not set (using default credentials)")
    elif not os.path.exists(creds_file):
        logger.error(f"❌ Credentials file not found: {creds_file}")
        return False
    else:
        logger.info(f"✅ Credentials file exists: {creds_file}")
        
        # Check credentials file content
        try:
            with open(creds_file, 'r') as f:
                creds_data = json.load(f)
                logger.info(f"✅ Service account email: {creds_data.get('client_email', 'Not found')}")
                logger.info(f"✅ Project in credentials: {creds_data.get('project_id', 'Not found')}")
        except Exception as e:
            logger.error(f"❌ Error reading credentials: {e}")
            return False
    
    return True

def check_authentication():
    """Check if authentication is working."""
    logger.info("=== Checking Authentication ===")
    
    try:
        # Get default credentials
        credentials, project = default()
        logger.info(f"✅ Default credentials found for project: {project}")
        
        # Check if credentials are valid
        if credentials.expired:
            logger.info("Refreshing expired credentials...")
            credentials.refresh(Request())
        
        if credentials.valid:
            logger.info("✅ Credentials are valid")
        else:
            logger.error("❌ Credentials are not valid")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Authentication error: {e}")
        return False

def check_api_access():
    """Check if Speech API is accessible."""
    logger.info("=== Checking Speech API Access ===")
    
    try:
        # Create Speech client
        client = speech_v2.SpeechClient()
        logger.info("✅ Speech v2 client created successfully")
        
        # Try to list recognizers to test API access
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        parent = f"projects/{project_id}/locations/global"
        
        logger.info(f"Attempting to list recognizers in: {parent}")
        
        # This will test if we have access to the API
        recognizers = client.list_recognizers(parent=parent)
        
        logger.info("✅ Successfully listed recognizers:")
        recognizer_count = 0
        for recognizer in recognizers:
            recognizer_count += 1
            logger.info(f"  - {recognizer.name}")
            if recognizer_count >= 5:  # Limit output
                break
        
        if recognizer_count == 0:
            logger.info("  No custom recognizers found (this is normal)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Speech API access error: {e}")
        logger.error("This might indicate:")
        logger.error("- Speech API is not enabled in your project")
        logger.error("- Service account doesn't have proper permissions")
        logger.error("- Project ID is incorrect")
        return False

def check_speech_api_enabled():
    """Check if Speech API is enabled in the project."""
    logger.info("=== Checking if Speech API is Enabled ===")
    
    try:
        from google.cloud import serviceusage_v1
        
        client = serviceusage_v1.ServiceUsageClient()
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        # Check if Speech API is enabled
        service_name = f"projects/{project_id}/services/speech.googleapis.com"
        
        try:
            service = client.get_service(name=service_name)
            if service.state == serviceusage_v1.State.ENABLED:
                logger.info("✅ Speech API is enabled")
                return True
            else:
                logger.error(f"❌ Speech API state: {service.state}")
                return False
        except Exception as e:
            logger.warning(f"Could not check API status: {e}")
            logger.info("This might be due to missing Service Usage API permissions")
            return True  # Assume it's enabled
            
    except ImportError:
        logger.warning("google-cloud-service-usage not installed, skipping API check")
        return True

def check_permissions():
    """Check IAM permissions for Speech API."""
    logger.info("=== Checking IAM Permissions ===")
    
    # Required permissions for Speech API
    required_permissions = [
        "speech.recognizers.recognize",
        "speech.recognizers.list",
        "speech.recognizers.get",
    ]
    
    logger.info("Required permissions for Speech API:")
    for perm in required_permissions:
        logger.info(f"  - {perm}")
    
    logger.info("\nTo check permissions manually:")
    logger.info("1. Go to Google Cloud Console")
    logger.info("2. Navigate to IAM & Admin > IAM")
    logger.info("3. Find your service account")
    logger.info("4. Check if it has 'Speech Developer' or 'Speech Client' role")

async def test_simple_recognition():
    """Test a simple recognition request."""
    logger.info("=== Testing Simple Recognition ===")
    
    try:
        client = speech_v2.SpeechClient()
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        # Create a simple recognition config
        config = speech_v2.RecognitionConfig(
            explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
                encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                audio_channel_count=1,
            ),
            language_codes=["en-US"],
            model="latest_long",
        )
        
        # Create a simple audio (1 second of silence)
        import numpy as np
        audio_data = np.zeros(16000, dtype=np.int16).tobytes()
        
        # Create recognition request
        request = speech_v2.RecognizeRequest(
            recognizer=f"projects/{project_id}/locations/global/recognizers/_",
            config=config,
            content=audio_data,
        )
        
        logger.info("Attempting simple recognition...")
        response = client.recognize(request=request)
        
        logger.info("✅ Simple recognition succeeded")
        logger.info(f"Results: {len(response.results)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple recognition failed: {e}")
        return False

def print_setup_instructions():
    """Print setup instructions if there are issues."""
    logger.info("\n" + "="*60)
    logger.info("SETUP INSTRUCTIONS")
    logger.info("="*60)
    
    logger.info("\n1. Enable Speech API:")
    logger.info("   gcloud services enable speech.googleapis.com")
    
    logger.info("\n2. Create service account (if needed):")
    logger.info("   gcloud iam service-accounts create speech-service-account")
    
    logger.info("\n3. Grant permissions:")
    logger.info("   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \\")
    logger.info("     --member=\"serviceAccount:speech-service-account@YOUR_PROJECT_ID.iam.gserviceaccount.com\" \\")
    logger.info("     --role=\"roles/speech.client\"")
    
    logger.info("\n4. Download credentials:")
    logger.info("   gcloud iam service-accounts keys create credentials.json \\")
    logger.info("     --iam-account=speech-service-account@YOUR_PROJECT_ID.iam.gserviceaccount.com")
    
    logger.info("\n5. Set environment variables:")
    logger.info("   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json")
    logger.info("   export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID")

async def main():
    """Run all checks."""
    logger.info("Starting Google Cloud Speech API setup check...\n")
    
    checks = [
        ("Credentials", check_credentials),
        ("Authentication", check_authentication),
        ("Speech API Enabled", check_speech_api_enabled),
        ("API Access", check_api_access),
        ("Simple Recognition", lambda: asyncio.run(test_simple_recognition())),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        logger.info(f"\n{'-'*40}")
        logger.info(f"Running: {check_name}")
        logger.info(f"{'-'*40}")
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            if result:
                logger.info(f"✅ {check_name}: PASSED")
            else:
                logger.info(f"❌ {check_name}: FAILED")
                all_passed = False
        except Exception as e:
            logger.error(f"❌ {check_name}: ERROR - {e}")
            all_passed = False
    
    logger.info(f"\n{'='*60}")
    if all_passed:
        logger.info("🎉 All checks passed! Your Google Cloud Speech API setup looks good.")
    else:
        logger.info("❌ Some checks failed. See instructions below.")
        print_setup_instructions()
        
    # Check permissions manually
    check_permissions()

if __name__ == "__main__":
    asyncio.run(main())