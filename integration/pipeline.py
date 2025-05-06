# Add these imports at the top of the file
import asyncio
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable, Awaitable, Set
import numpy as np

# Only updating the relevant parts of the file that need to change

class VoiceAIAgentPipeline:
    # ... (existing code) ...
    
    async def _process_audio(self, ws) -> None:
        """
        Process accumulated audio data through the pipeline with Google Cloud Speech.
        
        Args:
            ws: WebSocket connection
        """
        try:
            # Convert buffer to PCM with enhanced processing
            try:
                mulaw_bytes = bytes(self.input_buffer)
                
                # Convert using the enhanced audio processing
                pcm_audio = self.audio_processor.mulaw_to_pcm(mulaw_bytes)
                
            except Exception as e:
                logger.error(f"Error converting audio: {e}")
                # Clear part of buffer and try again next time
                half_size = len(self.input_buffer) // 2
                self.input_buffer = self.input_buffer[half_size:]
                return
                
            # Add some checks for audio quality
            if len(pcm_audio) < 1000:  # Very small audio chunk
                logger.warning(f"Audio chunk too small: {len(pcm_audio)} samples")
                return
            
            # Process audio through the STT pipeline
            try:
                # Convert to bytes format for Google Cloud Speech
                audio_bytes = (pcm_audio * 32767).astype(np.int16).tobytes()
                
                # Set agent speaking state for barge-in detection
                if hasattr(self.speech_recognizer, 'set_agent_speaking'):
                    self.speech_recognizer.set_agent_speaking(self.is_agent_responding)
                elif hasattr(self.stt_helper, 'set_agent_speaking'):
                    self.stt_helper.set_agent_speaking(self.is_agent_responding)
                
                # Define a callback to handle transcription results including barge-in
                async def process_result(result):
                    # Handle barge-in detection
                    if hasattr(result, 'barge_in_detected') and result.barge_in_detected:
                        logger.info("Barge-in detected! Interrupting agent response.")
                        self.interrupt_response = True
                        # Empty the output buffer to stop current speech
                        self.output_buffer.clear()
                        # Signal interruption to any ongoing processing
                        if hasattr(self, 'response_interrupted'):
                            self.response_interrupted.set()
                    
                    # Process transcription result
                    if hasattr(result, 'is_final') and result.is_final:
                        transcription = result.text
                        logger.info(f"Final transcription: {transcription}")
                        
                        # Process through knowledge base and generate response
                        # Only if not already processing and not interrupted
                        if not self.is_processing and not self.interrupt_response:
                            await self._process_transcription(transcription, ws)
                
                # Process audio chunk
                await self.speech_recognizer.process_audio_chunk(
                    audio_chunk=audio_bytes,
                    callback=process_result
                )
                
            except Exception as e:
                logger.error(f"Error processing audio through STT: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
    
    async def _process_transcription(self, transcription: str, ws) -> None:
        """
        Process a valid transcription through the knowledge base and TTS.
        
        Args:
            transcription: Validated transcription text
            ws: WebSocket connection
        """
        # Flag that we're processing to prevent duplicate processing
        self.is_processing = True
        self.interrupt_response = False
        
        try:
            # Create an event for interruption signaling
            self.response_interrupted = asyncio.Event()
            
            # Clean up transcription
            transcription = self.stt_helper.cleanup_transcription(transcription)
            
            # Only process if it's a valid transcription
            if transcription and self.stt_helper.is_valid_transcription(transcription):
                logger.info(f"Processing transcription: {transcription}")
                
                # Clear the input buffer since we have a valid transcription
                self.input_buffer.clear()
                
                # Don't process duplicate transcriptions
                if transcription == self.last_transcription:
                    logger.info("Duplicate transcription, not processing again")
                    self.is_processing = False
                    return
                
                # Process through knowledge base
                try:
                    # Set the agent as speaking during response generation and playback
                    self.is_agent_responding = True
                    
                    if hasattr(self.pipeline, 'query_engine'):
                        query_result = await self.pipeline.query_engine.query(transcription)
                        response = query_result.get("response", "")
                        
                        logger.info(f"Generated response: {response}")
                        
                        # Check if interrupted before sending response
                        if self.interrupt_response or self.response_interrupted.is_set():
                            logger.info("Response interrupted, not sending audio")
                            self.is_agent_responding = False
                            self.is_processing = False
                            return
                        
                        # Convert to speech with TTS
                        if response:
                            # Try to use TTS to generate speech
                            try:
                                speech_audio = await self.pipeline.tts_integration.text_to_speech(response)
                                logger.info(f"Generated speech: {len(speech_audio)} bytes")
                                                                
                                # Send back to client in smaller chunks for better interrupt handling
                                await self._send_audio_with_interrupt_check(speech_audio, ws)
                                
                                # Update state
                                self.last_transcription = transcription
                                self.last_response_time = time.time()
                                
                            except Exception as tts_error:
                                logger.error(f"Error with TTS: {tts_error}")
                                # Send a fallback text response
                                await self._send_text_fallback(response, ws)
                    
                except Exception as e:
                    logger.error(f"Error processing through knowledge base: {e}", exc_info=True)
                    
                    # Try to send a fallback response
                    await self._send_text_fallback(
                        "I'm sorry, I'm having trouble understanding. Could you try again?", 
                        ws
                    )
            else:
                logger.info(f"Invalid transcription: {transcription}")
        finally:
            # Reset processing state
            self.is_processing = False
            self.is_agent_responding = False
    
    async def _send_audio_with_interrupt_check(self, audio_data: bytes, ws) -> bool:
        """
        Send audio data to client with checks for interruptions.
        
        Args:
            audio_data: Audio data as bytes
            ws: WebSocket connection
            
        Returns:
            True if audio was sent successfully, False if interrupted
        """
        # Split audio into smaller chunks for better interrupt handling
        chunk_size = 4000  # Smaller chunks (~250ms of audio)
        chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
        
        logger.debug(f"Splitting {len(audio_data)} bytes into {len(chunks)} chunks")
        
        # Set agent speaking state for barge-in detection
        if hasattr(self.speech_recognizer, 'set_agent_speaking'):
            self.speech_recognizer.set_agent_speaking(True)
        elif hasattr(self.stt_helper, 'set_agent_speaking'):
            self.stt_helper.set_agent_speaking(True)
        
        try:
            for i, chunk in enumerate(chunks):
                # Check for interruption before sending each chunk
                if self.interrupt_response or (hasattr(self, 'response_interrupted') and self.response_interrupted.is_set()):
                    logger.info(f"Sending interrupted at chunk {i+1}/{len(chunks)}")
                    return False
                
                # Send the chunk
                await self._send_audio_chunk(chunk, ws)
                
                # Small delay between chunks to allow for interruptions
                await asyncio.sleep(0.01)
            
            return True
            
        finally:
            # Reset agent speaking state
            if hasattr(self.speech_recognizer, 'set_agent_speaking'):
                self.speech_recognizer.set_agent_speaking(False)
            elif hasattr(self.stt_helper, 'set_agent_speaking'):
                self.stt_helper.set_agent_speaking(False)
    
    async def _send_audio_chunk(self, chunk: bytes, ws) -> None:
        """
        Send a single audio chunk to the client.
        
        Args:
            chunk: Audio chunk as bytes
            ws: WebSocket connection
        """
        # Implement the audio sending logic here
        # This would depend on your specific implementation
        pass
    
    async def _send_text_fallback(self, text: str, ws) -> None:
        """
        Send a text fallback when audio generation fails.
        
        Args:
            text: Text to send
            ws: WebSocket connection
        """
        # Implement the fallback logic here
        # This would depend on your specific implementation
        pass