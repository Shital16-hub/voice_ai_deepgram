"""
Voice AI Agent main class that coordinates all components with STT integration.
Generic version that works with any knowledge base.
"""
import os
import logging
import asyncio
import time  # Add this import
from typing import Optional, Dict, Any, Union, Callable, Awaitable
import numpy as np
from scipy import signal

# Import only what's available and necessary
from speech_to_text.stt_integration import STTIntegration
from knowledge_base.conversation_manager import ConversationManager
from knowledge_base.llama_index.document_store import DocumentStore
from knowledge_base.llama_index.index_manager import IndexManager
from knowledge_base.llama_index.query_engine import QueryEngine
from text_to_speech import ElevenLabsTTS  # Updated import

logger = logging.getLogger(__name__)

class VoiceAIAgent:
    """Main Voice AI Agent class that coordinates all components."""
    
    def __init__(
        self,
        storage_dir: str = './storage',
        model_name: str = 'mistral:7b-instruct-v0.2-q4_0',
        api_key: Optional[str] = None,
        llm_temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Voice AI Agent.
        
        Args:
            storage_dir: Directory for persistent storage
            model_name: LLM model name for knowledge base
            api_key: API key for services (if needed)
            llm_temperature: LLM temperature for response generation
            **kwargs: Additional parameters for customization
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.api_key = api_key
        self.llm_temperature = llm_temperature
        
        # STT Parameters
        self.stt_language = kwargs.get('language', 'en-US')
        self.stt_keywords = kwargs.get('keywords', ['price', 'plan', 'cost', 'subscription', 'service'])
        
        # Component placeholders
        self.speech_recognizer = None
        self.stt_integration = None
        self.conversation_manager = None
        self.query_engine = None
        self.tts_client = None
        
        # Noise floor tracking for adaptive threshold
        self.noise_floor = 0.005
        self.noise_samples = []
        self.max_noise_samples = 20
        
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio for better speech recognition.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Processed audio data
        """
        try:
            # Update noise floor from quiet sections
            self._update_noise_floor(audio)
            
            # Apply high-pass filter to remove low-frequency noise
            b, a = signal.butter(6, 100/(16000/2), 'highpass')
            audio = signal.filtfilt(b, a, audio)
            
            # Apply band-pass filter for telephony frequency range
            b, a = signal.butter(4, [300/(16000/2), 3400/(16000/2)], 'band')
            audio = signal.filtfilt(b, a, audio)
            
            # Apply pre-emphasis to boost high frequencies
            audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
            
            # Apply noise gate with adaptive threshold
            threshold = self.noise_floor * 3.0
            audio = np.where(np.abs(audio) < threshold, 0, audio)
            
            # Normalize audio level
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio * (0.9 / max_val)
                
            return audio
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return audio  # Return original if processing fails
    
    def _update_noise_floor(self, audio: np.ndarray) -> None:
        """Update noise floor estimate from quiet sections."""
        # Find quiet sections (bottom 10% of energy)
        frame_size = min(len(audio), int(0.02 * 16000))  # 20ms frames
        if frame_size <= 1:
            return
            
        frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]
        frame_energies = [np.mean(np.square(frame)) for frame in frames]
        
        if len(frame_energies) > 0:
            # Sort energies and take bottom 10%
            sorted_energies = sorted(frame_energies)
            quiet_count = max(1, len(sorted_energies) // 10)
            quiet_energies = sorted_energies[:quiet_count]
            
            # Update noise samples
            self.noise_samples.extend(quiet_energies)
            
            # Limit sample count
            if len(self.noise_samples) > self.max_noise_samples:
                self.noise_samples = self.noise_samples[-self.max_noise_samples:]
            
            # Update noise floor with safety limits
            if self.noise_samples:
                self.noise_floor = max(
                    0.001,  # Minimum
                    min(0.02, np.percentile(self.noise_samples, 90) * 1.5)  # Maximum
                )
        
    async def init(self):
        """Initialize all components."""
        logger.info("Initializing Voice AI Agent components...")
        
        # We'll skip the speech recognizer initialization for now
        # Just initialize STT integration without a specific recognizer
        self.stt_integration = STTIntegration(
            speech_recognizer=None,
            language=self.stt_language
        )
        
        # Initialize document store and index manager
        doc_store = DocumentStore()
        index_manager = IndexManager(storage_dir=self.storage_dir)
        await index_manager.init()
        
        # Initialize query engine
        self.query_engine = QueryEngine(
            index_manager=index_manager, 
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature
        )
        await self.query_engine.init()
        
        # Initialize conversation manager with optimized parameters
        self.conversation_manager = ConversationManager(
            query_engine=self.query_engine,
            llm_model_name=self.model_name,
            llm_temperature=self.llm_temperature,
            # Skip greeting for better telephone experience
            skip_greeting=True
        )
        await self.conversation_manager.init()
        
        # Initialize TTS client with ElevenLabs
        self.tts_client = ElevenLabsTTS()
        
        logger.info("Voice AI Agent initialization complete with ElevenLabs TTS")
    
    async def process_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        callback: Optional[Callable[[Any], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process audio data.
        
        Args:
            audio_data: Audio data as numpy array or bytes
            callback: Optional callback function
            
        Returns:
            Processing result
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
        
        # Process audio for better speech recognition
        if isinstance(audio_data, np.ndarray):
            audio_data = self._process_audio(audio_data)
        
        # Since we don't have the specific STT implementation, we'll return a placeholder
        return {
            "status": "error",
            "error": "STT processing not implemented in this version"
        }
    
    async def process_streaming_audio(
        self,
        audio_stream,
        result_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process streaming audio with real-time response.
        
        Args:
            audio_stream: Async iterator of audio chunks
            result_callback: Callback for streaming results
            
        Returns:
            Final processing stats
        """
        if not self.initialized:
            raise RuntimeError("Voice AI Agent not initialized")
            
        # Track stats
        start_time = time.time()
        chunks_processed = 0
        results_count = 0
        
        # Return stats
        return {
            "status": "error",
            "error": "Streaming audio processing not implemented in this version",
            "chunks_processed": chunks_processed,
            "results_count": results_count,
            "total_time": time.time() - start_time
        }
    
    @property
    def initialized(self) -> bool:
        """Check if all components are initialized."""
        return (self.conversation_manager is not None and 
                self.query_engine is not None)
                
    async def shutdown(self):
        """Shut down all components properly."""
        logger.info("Shutting down Voice AI Agent...")
        
        # Reset conversation if active
        if self.conversation_manager:
            self.conversation_manager.reset()