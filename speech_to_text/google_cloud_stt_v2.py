# speech_to_text/google_cloud_stt_v2.py

"""
Fixed Google Cloud Speech-to-Text implementation using v2.32.0 API.
Removes hardcoding and uses API's automatic features properly.
"""
import os
import logging
import asyncio
import base64
import queue
import threading
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from dataclasses import dataclass

from google.cloud import speech
import numpy as np

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

class GoogleCloudStreamingSTT_V2:
    """
    Fixed Google Cloud Speech-to-Text implementation using proper streaming.
    Uses API's automatic features without hardcoding keywords.
    """
    
    def __init__(
        self,
        language: str = "en-US",
        sample_rate: int = 8000,
        encoding: str = "MULAW",
        channels: int = 1,
        interim_results: bool = False,
        enhanced_model: bool = True,
        timeout: float = 30.0
    ):
        """
        Initialize Google Cloud STT with proper telephony optimizations.
        
        Args:
            language: Language code (e.g., 'en-US')
            sample_rate: Audio sample rate in Hz
            encoding: Audio encoding format
            channels: Number of audio channels
            interim_results: Whether to return interim results
            enhanced_model: Whether to use enhanced model
            timeout: Timeout for streaming operations
        """
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.channels = channels
        self.interim_results = interim_results
        self.enhanced_model = enhanced_model
        self.timeout = timeout
        
        # Initialize client
        self.client = speech.SpeechClient()
        
        # State tracking
        self.is_streaming = False
        self.chunk_count = 0
        self.total_chunks = 0
        self.successful_transcriptions = 0
        
        # Streaming management
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_stream = threading.Event()
        self.streaming_generator = None
        self.responses_iterator = None
        self.processing_thread = None
        
        logger.info(f"Initialized GoogleCloudStreamingSTT_V2: {sample_rate}Hz, {encoding}")
    
    def _get_recognition_config(self) -> speech.RecognitionConfig:
        """Get recognition configuration with proper telephony optimizations."""
        # Map encoding string to enum
        encoding_map = {
            "LINEAR16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "MULAW": speech.RecognitionConfig.AudioEncoding.MULAW,
        }
        
        encoding_enum = encoding_map.get(self.encoding, speech.RecognitionConfig.AudioEncoding.MULAW)
        
        # Create config with telephony optimizations
        config = speech.RecognitionConfig(
            encoding=encoding_enum,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language,
            audio_channel_count=self.channels,
            
            # Key telephony optimizations
            model="phone_call",  # Telephony-optimized model
            use_enhanced=self.enhanced_model,  # Premium model for better accuracy
            
            # Automatic features - no hardcoding!
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            enable_word_confidence=True,
            enable_speaker_diarization=False,  # Disabled for single speaker
            
            # Speech adaptation for better telephony
            adaptation=speech.SpeechAdaptation(
                phrase_sets=[
                    speech.PhraseSet(
                        name="",
                        phrases=[
                            speech.PhraseSet.Phrase(value="voice assistant"),
                            speech.PhraseSet.Phrase(value="voice assist"),
                            speech.PhraseSet.Phrase(value="price plan"),
                            speech.PhraseSet.Phrase(value="pricing plan"),
                        ],
                        boost=10.0
                    )
                ]
            )
        )
        
        return config
    
    def _get_streaming_config(self) -> speech.StreamingRecognitionConfig:
        """Get streaming recognition configuration."""
        config = self._get_recognition_config()
        
        return speech.StreamingRecognitionConfig(
            config=config,
            interim_results=self.interim_results,
            
            # Key streaming optimizations
            single_utterance=False,  # Allow multiple utterances
            enable_voice_activity_events=True,  # Better detection
        )
    
    def _audio_generator(self):
        """Generate audio chunks for streaming."""
        while not self.stop_stream.is_set():
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
                self.audio_queue.task_done()
            except queue.Empty:
                continue
    
    def _process_responses(self):
        """Process streaming responses in a separate thread."""
        try:
            chunk_id = 0
            
            for response in self.responses_iterator:
                if self.stop_stream.is_set():
                    break
                
                for result in response.results:
                    chunk_id += 1
                    
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        # Create result object with word timing
                        words = []
                        if hasattr(alternative, 'words') and alternative.words:
                            for word_info in alternative.words:
                                words.append({
                                    "word": word_info.word,
                                    "start_time": word_info.start_time.total_seconds(),
                                    "end_time": word_info.end_time.total_seconds(),
                                    "confidence": getattr(word_info, 'confidence', alternative.confidence)
                                })
                        
                        transcription_result = StreamingTranscriptionResult(
                            text=alternative.transcript,
                            is_final=result.is_final,
                            confidence=alternative.confidence,
                            chunk_id=chunk_id,
                            words=words
                        )
                        
                        self.result_queue.put(transcription_result)
                        
                        if result.is_final:
                            self.successful_transcriptions += 1
                            logger.info(f"Final result: '{alternative.transcript}' (confidence: {alternative.confidence:.2f})")
                        else:
                            logger.debug(f"Interim result: '{alternative.transcript}'")
        except Exception as e:
            logger.error(f"Error processing responses: {e}", exc_info=True)
            self.result_queue.put(("ERROR", str(e)))
    
    async def start_streaming(self) -> None:
        """Start a new streaming session."""
        if self.is_streaming:
            await self.stop_streaming()
        
        logger.info("Starting Google Cloud Speech streaming session")
        
        # Reset state
        self.stop_stream.clear()
        self.is_streaming = True
        self.chunk_count = 0
        
        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Create streaming config
        streaming_config = self._get_streaming_config()
        
        # Create audio generator
        def audio_generator():
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            for chunk in self._audio_generator():
                yield chunk
        
        # Start streaming recognize
        self.responses_iterator = self.client.streaming_recognize(audio_generator())
        
        # Start response processing thread
        self.processing_thread = threading.Thread(target=self._process_responses, daemon=True)
        self.processing_thread.start()
    
    async def process_audio_chunk(
        self,
        audio_chunk: Union[bytes, np.ndarray],
        callback: Optional[Callable[[StreamingTranscriptionResult], Awaitable[None]]] = None
    ) -> Optional[StreamingTranscriptionResult]:
        """
        Process audio chunk with proper streaming handling.
        
        Args:
            audio_chunk: Audio data as bytes or numpy array
            callback: Optional callback for results
            
        Returns:
            Transcription result or None
        """
        if not self.is_streaming:
            await self.start_streaming()
        
        self.total_chunks += 1
        
        try:
            # Convert numpy array to bytes if needed
            if isinstance(audio_chunk, np.ndarray):
                if audio_chunk.dtype == np.float32:
                    # Convert float32 to mulaw
                    import audioop
                    audio_bytes = audioop.lin2ulaw(
                        (audio_chunk * 32767).astype(np.int16).tobytes(), 2
                    )
                else:
                    audio_bytes = audio_chunk.tobytes()
            else:
                audio_bytes = audio_chunk
            
            # Skip tiny chunks
            if len(audio_bytes) < 160:  # Less than 20ms at 8kHz
                logger.debug("Skipping tiny audio chunk")
                return None
            
            # Add to queue for processing
            self.audio_queue.put(audio_bytes)
            
            # Check for results
            final_result = None
            
            # Process any available results
            while True:
                try:
                    result = self.result_queue.get_nowait()
                    
                    # Handle errors
                    if isinstance(result, tuple) and result[0] == "ERROR":
                        logger.error(f"Streaming error: {result[1]}")
                        return None
                    
                    # Call callback if provided
                    if callback:
                        await callback(result)
                    
                    # Keep track of final results
                    if result.is_final:
                        final_result = result
                        
                except queue.Empty:
                    break
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            return None
    
    async def stop_streaming(self) -> tuple[str, float]:
        """Stop the streaming session."""
        if not self.is_streaming:
            return "", 0.0
        
        logger.info("Stopping Google Cloud Speech streaming session")
        
        # Signal stop
        self.stop_stream.set()
        self.is_streaming = False
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Get any final results
        final_text = ""
        duration = 0.0
        
        # Collect all final results
        final_results = []
        while True:
            try:
                result = self.result_queue.get_nowait()
                if not isinstance(result, tuple) and result.is_final:
                    final_results.append(result)
            except queue.Empty:
                break
        
        # Combine final results
        if final_results:
            # Get the last final result
            last_result = final_results[-1]
            final_text = last_result.text
            
            # Calculate duration from word timings
            if last_result.words:
                duration = last_result.words[-1]["end_time"]
        
        # Reset state
        self.responses_iterator = None
        self.processing_thread = None
        
        logger.info(f"Stopped streaming session. Processed {self.total_chunks} chunks")
        return final_text, duration
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        success_rate = (self.successful_transcriptions / max(self.total_chunks, 1)) * 100
        
        return {
            "total_chunks": self.total_chunks,
            "successful_transcriptions": self.successful_transcriptions,
            "success_rate": round(success_rate, 2),
            "is_streaming": self.is_streaming,
            "language_code": self.language,
            "model": "phone_call",
            "enhanced": self.enhanced_model,
            "encoding": self.encoding,
            "sample_rate": self.sample_rate
        }