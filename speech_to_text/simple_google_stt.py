"""
Optimized Google Cloud Speech client with direct MULAW support.
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
import numpy as np
from dataclasses import dataclass

try:
    from google.cloud import speech
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

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
    """Google Cloud Speech client optimized for Twilio with direct MULAW support."""
    
    def __init__(
        self, 
        language="en-US", 
        sample_rate=8000,  # Changed to 8000 for direct MULAW support
        encoding="MULAW",  # Changed to MULAW
        channels=1, 
        interim_results=True,
        enhanced_model=True,
        speech_context_phrases=None
    ):
        """Initialize the Google Cloud Speech client for MULAW support."""
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud Speech API not available")
        
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.enhanced_model = enhanced_model
        self.speech_context_phrases = speech_context_phrases or [
            "pricing", "plan", "cost", "subscription", "service", "features",
            "support", "upgrade", "payment", "account", "question", "help"
        ]
        
        # Create client
        try:
            self.client = speech.SpeechClient()
            logger.info("Google Cloud Speech client created successfully with MULAW support")
        except Exception as e:
            logger.error(f"Failed to create Google Cloud Speech client: {e}")
            raise
        
        # State tracking
        self.is_streaming = False
        self.utterance_id = 0
        self._stream = None
        self._config = None
    
    def _get_config(self) -> speech.RecognitionConfig:
        """Get the recognition config optimized for MULAW telephony."""
        try:
            # Get encoding enum
            if self.encoding == "MULAW":
                encoding_enum = speech.RecognitionConfig.AudioEncoding.MULAW
            else:
                encoding_enum = speech.RecognitionConfig.AudioEncoding.LINEAR16
            
            config = speech.RecognitionConfig(
                encoding=encoding_enum,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language,
                enable_automatic_punctuation=True,
                use_enhanced=self.enhanced_model,
                model="telephony",  # Optimized for phone calls
                audio_channel_count=self.channels,
                enable_word_time_offsets=True,
                profanity_filter=False,
            )
            
            # Add speech contexts for better recognition
            if self.speech_context_phrases:
                speech_context = speech.SpeechContext(
                    phrases=self.speech_context_phrases,
                    boost=15.0  # High boost for telephony
                )
                config.speech_contexts.append(speech_context)
            
            return config
        except Exception as e:
            logger.error(f"Error creating recognition config: {e}")
            raise
    
    def _get_streaming_config(self) -> speech.StreamingRecognitionConfig:
        """Get the streaming recognition config."""
        recognition_config = self._get_config()
        
        return speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=self.interim_results,
            single_utterance=False
        )
    
    async def start_streaming(self):
        """Start streaming session with proper MULAW support."""
        try:
            if self.is_streaming:
                await self.stop_streaming()
            
            self._config = self._get_streaming_config()
            
            # Create a generator for the audio stream
            self._audio_generator = self._create_audio_generator()
            
            # Start the streaming recognize
            self._stream = self.client.streaming_recognize(
                requests=self._audio_generator,
                retry=speech.types.Retry(
                    predicate=speech.types.Retry.if_transient_error,
                    initial=1.0,
                    maximum=60.0,
                    multiplier=2.0
                )
            )
            
            self.is_streaming = True
            self.utterance_id = 0
            logger.info("Started Google Cloud Speech streaming session with MULAW")
        except Exception as e:
            logger.error(f"Error starting streaming session: {e}")
            self.is_streaming = False
            raise
    
    def _create_audio_generator(self):
        """Create audio generator for streaming."""
        # Send the config in the first request
        yield speech.StreamingRecognizeRequest(streaming_config=self._config)
        
        # Then yield audio content as we receive it
        while self.is_streaming:
            # This generator will be fed with audio data
            chunk = yield
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop streaming session and get final result."""
        if not self.is_streaming:
            return "", 0.0
        
        try:
            self.is_streaming = False
            
            # Close the generator
            if hasattr(self, '_audio_generator'):
                try:
                    self._audio_generator.close()
                except:
                    pass
            
            # Get any final results from the stream
            final_text = ""
            duration = 0.0
            
            if self._stream:
                try:
                    # Process any remaining responses
                    for response in self._stream:
                        for result in response.results:
                            if result.is_final and result.alternatives:
                                final_text = result.alternatives[0].transcript
                                if result.alternatives[0].words:
                                    duration = (result.alternatives[0].words[-1].end_time.total_seconds() - 
                                              result.alternatives[0].words[0].start_time.total_seconds())
                                break
                except:
                    pass
            
            logger.info(f"Stopped Google Cloud Speech streaming session. Final text: '{final_text}'")
            return final_text, duration
        except Exception as e:
            logger.error(f"Error stopping streaming session: {e}")
            return "", 0.0
    
    async def process_audio_chunk(
        self, 
        audio_chunk, 
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ):
        """Process a chunk of audio with optimized MULAW handling."""
        if not self.is_streaming:
            logger.warning("Called process_audio_chunk but streaming is not active")
            await self.start_streaming()
        
        try:
            # For MULAW, we can send the data directly
            if isinstance(audio_chunk, np.ndarray):
                if self.encoding == "MULAW":
                    # Convert float32 array back to MULAW bytes
                    audio_bytes = (audio_chunk * 127).astype(np.uint8).tobytes()
                else:
                    # Convert to int16 for LINEAR16
                    audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            else:
                audio_bytes = audio_chunk
            
            # Check minimum chunk size
            if len(audio_bytes) < 320:  # Less than 40ms at 8kHz
                logger.debug(f"Audio chunk too small: {len(audio_bytes)} bytes")
                return None
            
            # Send audio to the stream
            try:
                # Send audio data to the generator
                self._audio_generator.send(audio_bytes)
                
                # Process any available responses
                result = None
                try:
                    for response in self._stream:
                        for speech_result in response.results:
                            if speech_result.alternatives:
                                alt = speech_result.alternatives[0]
                                
                                # Create result object
                                self.utterance_id += 1
                                transcription_result = StreamingTranscriptionResult(
                                    text=alt.transcript,
                                    is_final=speech_result.is_final,
                                    confidence=alt.confidence if hasattr(alt, "confidence") else 0.8,
                                    chunk_id=self.utterance_id
                                )
                                
                                # Call callback if provided
                                if callback:
                                    await callback(transcription_result)
                                
                                # Return final results
                                if speech_result.is_final:
                                    result = transcription_result
                                    break
                        
                        # Only process one response per chunk
                        break
                except Exception as e:
                    logger.debug(f"No response available yet: {e}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error sending audio to stream: {e}")
                # Try to restart the stream
                await self.stop_streaming()
                await self.start_streaming()
                return None
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    async def transcribe_file(
        self,
        file_path: str,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Transcribe an audio file."""
        try:
            # Read audio file based on encoding
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Create RecognitionAudio
            audio = speech.RecognitionAudio(content=content)
            
            # Get config
            config = self._get_config()
            
            # Perform recognition
            response = self.client.recognize(config=config, audio=audio)
            
            # Process results
            transcription = ""
            confidence = 0.0
            
            for result in response.results:
                for alt in result.alternatives:
                    transcription += alt.transcript + " "
                    confidence = max(confidence, alt.confidence)
            
            transcription = transcription.strip()
            
            return {
                "transcription": transcription,
                "confidence": confidence,
                "is_final": True
            }
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            return {
                "error": str(e),
                "transcription": "",
                "confidence": 0.0,
                "is_final": True
            }

# Alias for backward compatibility
class SimpleGoogleSTT(GoogleCloudStreamingSTT):
    """Alias for backward compatibility."""
    pass