"""
Google Cloud Speech-to-Text client for Voice AI Agent.
"""
import os
import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
import numpy as np
import re
from dataclasses import dataclass

try:
    from google.cloud import speech
    from google.api_core.exceptions import GoogleAPIError
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    
logger = logging.getLogger(__name__)

@dataclass
class StreamingTranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float
    start_time: float = 0.0
    end_time: float = 0.0
    chunk_id: int = 0
    words: List[Dict[str, Any]] = None


class GoogleCloudStreamingSTT:
    """
    Google Cloud Speech-to-Text streaming client for Voice AI Agent.

    This class handles real-time streaming Speech-to-Text conversion
    using Google Cloud's Speech API, optimized for telephony applications.
    """
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 16000,
        encoding: str = "LINEAR16",
        channels: int = 1,
        interim_results: bool = True,
        speech_context_phrases: Optional[List[str]] = None,
        enhanced_model: bool = True
    ):
        """
        Initialize the Google Cloud STT client.
        
        Args:
            language: Language code (BCP-47)
            sample_rate: Audio sample rate in Hz
            encoding: Audio encoding format
            channels: Number of audio channels
            interim_results: Whether to return interim results
            speech_context_phrases: Phrases to boost recognition
            enhanced_model: Whether to use enhanced telephony model
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud Speech modules not installed. Please run: pip install google-cloud-speech")

        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.speech_context_phrases = speech_context_phrases or [
            "pricing", "plan", "cost", "subscription", "service", "features",
            "support", "upgrade", "payment", "account", "question", "help"
        ]
        self.enhanced_model = enhanced_model
        
        # State management
        self.is_streaming = False
        self.stream = None
        self.streaming_client = None
        self.streaming_config = None
        self.utterance_id = 0
        self.last_result = None
        
        # Create the client
        try:
            self.client = speech.SpeechClient()
            logger.info("Initialized Google Cloud Speech-to-Text client")
        except Exception as e:
            logger.error(f"Error initializing Google Cloud Speech-to-Text client: {e}")
            raise
    
    def _get_recognition_config(self) -> speech.RecognitionConfig:
        """Get recognition configuration for Google Cloud Speech API."""
        # Get audio encoding enum
        encoding_enum = getattr(speech.RecognitionConfig.AudioEncoding, self.encoding)
        
        # Create RecognitionConfig
        config = speech.RecognitionConfig(
            encoding=encoding_enum,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            profanity_filter=False,
            model="telephony" if self.enhanced_model else "default",
            use_enhanced=self.enhanced_model,
        )
        
        # Add speech contexts for better recognition
        if self.speech_context_phrases:
            speech_context = speech.SpeechContext(
                phrases=self.speech_context_phrases,
                boost=15.0  # Boost these phrases
            )
            config.speech_contexts.append(speech_context)
            
        return config
    
    def _get_streaming_config(self) -> speech.StreamingRecognitionConfig:
        """Get streaming configuration for Google Cloud Speech API."""
        recognition_config = self._get_recognition_config()
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=self.interim_results,
            single_utterance=False
        )
        
        return streaming_config
    
    async def start_streaming(self) -> None:
        """Start a new streaming recognition session."""
        if self.is_streaming:
            await self.stop_streaming()
        
        # Create streaming client
        self.streaming_client = speech.SpeechClient()
        
        # Create streaming config
        self.streaming_config = self._get_streaming_config()
        
        # Create a request generator
        def request_generator():
            # First, yield the streaming config
            yield speech.StreamingRecognizeRequest(streaming_config=self.streaming_config)
            
            # This is a generator that will be used to send audio data later
            while True:
                # This will block until audio data is available
                data = yield
                if data is None:
                    break
                
                # Yield audio content
                yield speech.StreamingRecognizeRequest(audio_content=data)
        
        # Initialize generator
        self.request_generator = request_generator()
        next(self.request_generator)  # Prime the generator
        
        # Create bidirectional streaming RPC
        self.stream = self.streaming_client.streaming_recognize(
            requests=self.request_generator
        )
        
        # Reset state
        self.utterance_id = 0
        self.last_result = None
        self.is_streaming = True
        
        logger.info("Started Google Cloud STT streaming session")
    
    async def stop_streaming(self) -> tuple[str, float]:
        """
        Stop the streaming session and return final text.
        
        Returns:
            Tuple of (final_text, duration)
        """
        if not self.is_streaming:
            return "", 0.0
        
        final_text = ""
        duration = 0.0
        
        try:
            # Close the stream by sending None to the generator
            if self.request_generator:
                try:
                    self.request_generator.send(None)
                except (StopIteration, ValueError):
                    pass
            
            # Get the final result if available
            if self.last_result:
                final_text = self.last_result.text
                duration = self.last_result.end_time - self.last_result.start_time
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
        finally:
            self.is_streaming = False
            self.stream = None
            self.request_generator = None
            logger.info(f"Stopped Google Cloud STT streaming session. Final text: '{final_text}'")
        
        return final_text, duration
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process a chunk of streaming audio.
        
        Args:
            audio_chunk: Audio chunk as bytes or numpy array
            callback: Optional async callback for results
            
        Returns:
            Transcription result or None
        """
        if not self.is_streaming or not self.stream or not self.request_generator:
            logger.warning("Not streaming - call start_streaming() first")
            return None
        
        try:
            # Ensure audio_chunk is bytes
            if isinstance(audio_chunk, np.ndarray):
                # Convert float32 [-1.0, 1.0] to int16
                audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            else:
                audio_bytes = audio_chunk
            
            # Send audio chunk
            try:
                self.request_generator.send(audio_bytes)
            except StopIteration:
                # If generator has stopped, restart the streaming session
                logger.warning("Request generator stopped, restarting streaming session")
                await self.start_streaming()
                if not self.is_streaming:
                    return None
                    
                # Try sending audio again
                self.request_generator.send(audio_bytes)
            
            # Process responses (non-blocking)
            loop = asyncio.get_event_loop()
            
            # Function to process responses in a separate thread
            def process_responses():
                results = []
                try:
                    for response in self.stream.responses:
                        # Only process if still streaming
                        if not self.is_streaming:
                            break
                            
                        # Process each result in the response
                        for result in response.results:
                            is_final = result.is_final
                            
                            # Get alternatives
                            if result.alternatives:
                                alt = result.alternatives[0]
                                text = alt.transcript
                                confidence = alt.confidence if alt.confidence > 0 else 0.8  # Default if not provided
                                
                                # Extract word timings if available
                                words = []
                                start_time = 0.0
                                end_time = 0.0
                                
                                if hasattr(alt, 'words') and alt.words:
                                    for word in alt.words:
                                        start_sec = word.start_time.seconds + word.start_time.nanos / 1e9
                                        end_sec = word.end_time.seconds + word.end_time.nanos / 1e9
                                        
                                        # Add to words list
                                        words.append({
                                            "word": word.word,
                                            "start_time": start_sec,
                                            "end_time": end_sec,
                                            "confidence": confidence
                                        })
                                    
                                    # Set start and end times from first and last word
                                    if words:
                                        start_time = words[0]["start_time"]
                                        end_time = words[-1]["end_time"]
                                
                                # Create result object
                                self.utterance_id += 1
                                transcription_result = StreamingTranscriptionResult(
                                    text=text,
                                    is_final=is_final,
                                    confidence=confidence,
                                    start_time=start_time,
                                    end_time=end_time,
                                    chunk_id=self.utterance_id,
                                    words=words if words else None
                                )
                                
                                # Save if final
                                if is_final:
                                    self.last_result = transcription_result
                                
                                # Add to results
                                results.append(transcription_result)
                                
                                # Call callback if provided
                                if callback and loop.is_running():
                                    asyncio.run_coroutine_threadsafe(
                                        callback(transcription_result), 
                                        loop
                                    )
                                    
                except Exception as e:
                    logger.error(f"Error processing response stream: {e}")
                finally:
                    return results
            
            # Run in a separate thread to avoid blocking
            future = loop.run_in_executor(None, process_responses)
            
            # Wait a very short time for results, but don't block for too long
            try:
                results = await asyncio.wait_for(future, timeout=0.05)
                # Return the latest result if available
                if results:
                    return results[-1]
            except asyncio.TimeoutError:
                # No results yet, that's okay
                pass
            
            return None
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    async def transcribe_file(
        self,
        file_path: str,
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to audio file
            callback: Optional callback for interim results
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Read audio file
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Create RecognitionAudio
            audio = speech.RecognitionAudio(content=content)
            
            # Get config
            config = self._get_recognition_config()
            
            # Perform synchronous recognition
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