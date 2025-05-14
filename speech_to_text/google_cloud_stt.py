"""
Optimized Google Cloud Speech-to-Text v2 implementation for Twilio telephony.
"""
import logging
import asyncio
import time
import os
import json
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
import numpy as np
from dataclasses import dataclass

# Import Speech-to-Text v2 API
from google.cloud import speech_v2
from google.api_core.client_options import ClientOptions

logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    chunk_id: int = 0
    words: List[Dict[str, Any]] = None

class GoogleCloudStreamingSTT:
    """
    Google Cloud Speech-to-Text v2 client optimized for telephony with Twilio.
    Uses MULAW encoding at 8kHz for direct compatibility with Twilio.
    """
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = False,
        project_id: Optional[str] = None,
        location: str = "global",
        enhanced_model: bool = True
    ):
        """
        Initialize with optimal settings for Twilio telephony.
        """
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.enhanced_model = enhanced_model
        self.location = location
        
        # Get project ID with automatic extraction
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        # If not provided, try to extract from credentials file
        if not self.project_id:
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_file and os.path.exists(credentials_file):
                try:
                    with open(credentials_file, 'r') as f:
                        creds_data = json.load(f)
                        self.project_id = creds_data.get('project_id')
                        logger.info(f"Auto-extracted project ID from credentials: {self.project_id}")
                except Exception as e:
                    logger.error(f"Error reading credentials file: {e}")
        
        if not self.project_id:
            raise ValueError(
                "Google Cloud project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or ensure your credentials file contains a project_id field."
            )
        
        # Initialize v2 client with proper endpoint
        endpoint = f"{location}-speech.googleapis.com" if location != "global" else None
        client_options = ClientOptions(api_endpoint=endpoint) if endpoint else None
        
        self.client = speech_v2.SpeechClient(client_options=client_options)
        
        # Construct recognizer path
        self.recognizer_path = self.client.recognizer_path(
            project=self.project_id,
            location=location,
            recognizer="_"  # Use the default recognizer
        )
        
        # State tracking
        self.is_streaming = False
        self.chunk_count = 0
        self.total_chunks = 0
        self.successful_transcriptions = 0
        self.streaming_call = None
        self.request_queue = asyncio.Queue()
        
        logger.info(f"Initialized GoogleCloudStreamingSTT v2: {sample_rate}Hz, {encoding}, telephony-optimized")
        logger.info(f"Using project: {self.project_id}, location: {location}")
    
    def _get_recognition_config(self) -> speech_v2.RecognitionConfig:
        """Get optimized v2 recognition configuration for telephony."""
        
        # Create explicit decoding config for MULAW
        decoding_config = speech_v2.ExplicitDecodingConfig(
            encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.MULAW,
            sample_rate_hertz=self.sample_rate,
            audio_channel_count=self.channels,
        )
        
        # Create recognition features
        features = speech_v2.RecognitionFeatures(
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            enable_word_confidence=True,
        )
        
        # Create recognition config optimized for telephony
        config = speech_v2.RecognitionConfig(
            explicit_decoding_config=decoding_config,
            language_codes=[self.language],
            model="telephony" if self.enhanced_model else "latest_short",
            features=features,
        )
        
        return config
    
    async def start_streaming(self) -> None:
        """Start a new streaming session."""
        if self.is_streaming:
            await self.stop_streaming()
        
        self.is_streaming = True
        self.chunk_count = 0
        logger.info("Started Google Cloud Speech v2 streaming session")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop the streaming session."""
        if not self.is_streaming:
            return "", 0.0
        
        self.is_streaming = False
        
        # Close streaming call if active
        if self.streaming_call:
            try:
                await self.streaming_call.close()
            except Exception as e:
                logger.error(f"Error closing streaming call: {e}")
            self.streaming_call = None
        
        logger.info(f"Stopped streaming session. Processed {self.chunk_count} chunks")
        return "", 0.0
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process audio chunk with v2 API - batch processing for better accuracy.
        """
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        
        try:
            # Convert numpy array to bytes if needed
            if isinstance(audio_chunk, np.ndarray):
                # For MULAW, convert float32 to mulaw bytes
                if audio_chunk.dtype == np.float32:
                    import audioop
                    # Convert to 16-bit PCM first
                    pcm_data = (audio_chunk * 32767).astype(np.int16).tobytes()
                    audio_bytes = audioop.lin2ulaw(pcm_data, 2)
                else:
                    audio_bytes = audio_chunk.tobytes()
            else:
                audio_bytes = audio_chunk
            
            logger.debug(f"Processing audio chunk #{self.total_chunks}: {len(audio_bytes)} bytes")
            
            # Skip tiny chunks
            if len(audio_bytes) < 160:  # Less than 20ms at 8kHz
                logger.debug("Skipping tiny audio chunk")
                return None
            
            # Use batch recognition for better accuracy
            config = self._get_recognition_config()
            
            # Create recognition request
            request = speech_v2.RecognizeRequest(
                recognizer=self.recognizer_path,
                config=config,
                content=audio_bytes,
            )
            
            # Perform synchronous recognition
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.recognize(request=request)
            )
            
            # Process results
            if response.results:
                for result in response.results:
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        # Create result object
                        self.chunk_count += 1
                        transcription_result = StreamingTranscriptionResult(
                            text=alternative.transcript,
                            is_final=True,  # Batch results are always final
                            confidence=getattr(alternative, 'confidence', 0.9),
                            chunk_id=self.chunk_count
                        )
                        
                        self.successful_transcriptions += 1
                        logger.info(f"Transcription: '{alternative.transcript}' (confidence: {transcription_result.confidence:.2f})")
                        
                        # Call callback if provided
                        if callback:
                            await callback(transcription_result)
                        
                        return transcription_result
            
            logger.debug("No transcription results from Google Cloud Speech v2")
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        success_rate = (self.successful_transcriptions / max(self.total_chunks, 1)) * 100
        
        return {
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "success_rate": round(success_rate, 2),
            "is_streaming": self.is_streaming,
            "language_code": self.language,
            "model": "telephony" if self.enhanced_model else "latest_short",
            "api_version": "v2",
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
            "project_id": self.project_id
        }