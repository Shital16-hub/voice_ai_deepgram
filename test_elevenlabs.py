import requests
import os
import json
import asyncio
import time
import io
import numpy as np
from scipy import signal

async def test_voice_for_telephony(api_key, voice_id, model_id, test_text):
    """Test a specific voice and model for telephony use"""
    print(f"Testing voice {voice_id} with model {model_id}...")
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Prepare request data optimized for telephony
    data = {
        "model_id": model_id,
        "text": test_text,
        "voice_settings": {
            "stability": 0.75,  # Higher stability for consistent telephony
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    
    # Measure latency
    start_time = time.time()
    
    # Test standard synthesis
    try:
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            json=data,
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {
                "status": "error",
                "error": response.text,
                "voice_id": voice_id,
                "model_id": model_id
            }
        
        standard_latency = time.time() - start_time
        audio_data = response.content
        audio_size = len(audio_data)
        
        # Measure frequency characteristics (important for telephony)
        frequency_metrics = analyze_audio_frequencies(audio_data)
        
        # Test streaming latency
        streaming_start = time.time()
        
        # Make streaming request with small timeout
        streaming_response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
            json=data,
            headers=headers,
            stream=True
        )
        
        # Time to first byte
        first_chunk_received = False
        for chunk in streaming_response.iter_content(chunk_size=1024):
            if chunk:
                if not first_chunk_received:
                    first_chunk_received = True
                    first_chunk_time = time.time()
                    streaming_latency = first_chunk_time - streaming_start
                    break
        
        # Close connection
        streaming_response.close()
        
        return {
            "status": "success",
            "voice_id": voice_id,
            "model_id": model_id,
            "standard_latency": standard_latency,
            "streaming_latency": streaming_latency,
            "audio_size": audio_size,
            "frequency_metrics": frequency_metrics
        }
        
    except Exception as e:
        print(f"Error testing voice: {e}")
        return {
            "status": "error",
            "error": str(e),
            "voice_id": voice_id,
            "model_id": model_id
        }

def analyze_audio_frequencies(audio_data):
    """Analyze audio data for telephony frequency characteristics"""
    try:
        # Check if we can parse as WAV
        try:
            with io.BytesIO(audio_data) as buf:
                import wave
                with wave.open(buf, 'rb') as wav:
                    n_channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()
                    framerate = wav.getframerate()
                    n_frames = wav.getnframes()
                    audio_bytes = wav.readframes(n_frames)
                    
                    if sample_width == 2:  # 16 bit
                        dtype = np.int16
                    elif sample_width == 4:  # 32 bit
                        dtype = np.int32
                    else:
                        dtype = np.uint8
                        
                    audio_array = np.frombuffer(audio_bytes, dtype=dtype)
                    if n_channels > 1:
                        audio_array = audio_array.reshape(-1, n_channels)
                        audio_array = np.mean(audio_array, axis=1)  # Convert to mono
        except:
            # If WAV parsing fails, assume MP3 and use simple metrics
            return {
                "format": "mp3",
                "telephony_suitability": 0.8  # Default to good for MP3
            }
        
        # Normalize audio
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array.astype(np.float32) / np.max(np.abs(audio_array))
        
        # Calculate telephony-specific metrics
        # Telephony uses 300-3400 Hz frequency range
        if len(audio_array) > 0:
            # Compute FFT
            fft_result = np.fft.rfft(audio_array)
            fft_freqs = np.fft.rfftfreq(len(audio_array), 1/framerate)
            fft_magnitude = np.abs(fft_result)
            
            # Check energy in telephony range (300-3400 Hz)
            telephony_mask = (fft_freqs >= 300) & (fft_freqs <= 3400)
            telephony_energy = np.sum(fft_magnitude[telephony_mask])
            total_energy = np.sum(fft_magnitude)
            
            telephony_ratio = telephony_energy / total_energy if total_energy > 0 else 0
            
            # Calculate speech clarity metrics
            clarity_score = 0.0
            
            # Simple spectral flatness as a proxy for speech clarity
            # Lower values indicate more speech-like sounds (less noise)
            log_spectrum = np.log(fft_magnitude + 1e-10)
            spectral_flatness = np.exp(np.mean(log_spectrum)) / np.mean(fft_magnitude)
            
            # Convert to clarity score (higher is better)
            clarity_score = 1.0 - min(1.0, spectral_flatness * 10)
            
            return {
                "format": "wav",
                "sample_rate": framerate,
                "telephony_ratio": float(telephony_ratio),
                "clarity_score": float(clarity_score),
                "duration": n_frames / framerate,
                "telephony_suitability": float(telephony_ratio * clarity_score)
            }
        else:
            return {
                "format": "wav",
                "sample_rate": framerate,
                "error": "Empty audio data",
                "telephony_suitability": 0.0
            }
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return {"error": str(e), "telephony_suitability": 0.0}

async def main():
    # Get API key
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        api_key = input("Enter your ElevenLabs API key: ")
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Get models
    print("Fetching ElevenLabs models...")
    models_response = requests.get(
        "https://api.elevenlabs.io/v1/models",
        headers=headers
    )
    
    if models_response.status_code != 200:
        print(f"Error fetching models: {models_response.status_code}")
        print(models_response.text)
        return
    
    models_data = models_response.json()
    models = models_data if isinstance(models_data, list) else models_data.get("models", [])
    
    print(f"Found {len(models)} models")
    for i, model in enumerate(models):
        print(f"{i+1}. {model.get('name')} (ID: {model.get('model_id')})")
    
    # Get voices
    print("\nFetching ElevenLabs voices...")
    voices_response = requests.get(
        "https://api.elevenlabs.io/v1/voices",
        headers=headers
    )
    
    if voices_response.status_code != 200:
        print(f"Error fetching voices: {voices_response.status_code}")
        print(voices_response.text)
        return
    
    voices_data = voices_response.json()
    voices = voices_data.get("voices", [])
    
    print(f"Found {len(voices)} voices")
    for i, voice in enumerate(voices):
        print(f"{i+1}. {voice.get('name')} (ID: {voice.get('voice_id')})")
    
    # Telephony test text
    test_text = "Welcome to our customer service line. How may I assist you today?"
    
    # Select up to 3 models to test
    max_models = min(3, len(models))
    models_to_test = models[:max_models]
    
    # Select up to 5 voices to test
    max_voices = min(5, len(voices))
    voices_to_test = voices[:max_voices]
    
    print(f"\nTesting {len(models_to_test)} models with {len(voices_to_test)} voices for telephony use...")
    
    # Run tests
    results = []
    
    for model in models_to_test:
        model_id = model.get('model_id')
        model_name = model.get('name')
        
        for voice in voices_to_test:
            voice_id = voice.get('voice_id')
            voice_name = voice.get('name')
            
            print(f"Testing model '{model_name}' with voice '{voice_name}'...")
            
            result = await test_voice_for_telephony(api_key, voice_id, model_id, test_text)
            
            # Add names for easier reading
            result['model_name'] = model_name
            result['voice_name'] = voice_name
            
            if result.get('status') == 'success':
                print(f"Test successful - Latency: {result.get('standard_latency', 0):.2f}s, " 
                      f"Streaming latency: {result.get('streaming_latency', 0):.2f}s")
            else:
                print(f"Test failed: {result.get('error', 'Unknown error')}")
            
            results.append(result)
    
    # Rank results
    print("\n===== TELEPHONY SUITABILITY RANKING =====")
    
    # Define scoring function
    def telephony_score(result):
        if result.get('status') != 'success':
            return -1000
        
        # Factors with weights:
        # 1. Streaming latency (50%) - lower is better
        # 2. Telephony suitability (30%) - higher is better
        # 3. Standard latency (20%) - lower is better
        
        streaming_latency = result.get('streaming_latency', 10)
        streaming_score = max(0, 10 - streaming_latency * 5)  # 0-10 points
        
        telephony_suit = result.get('frequency_metrics', {}).get('telephony_suitability', 0)
        telephony_score = telephony_suit * 10  # 0-10 points
        
        std_latency = result.get('standard_latency', 10)
        std_latency_score = max(0, 10 - std_latency * 2)  # 0-10 points
        
        total = (streaming_score * 0.5) + (telephony_score * 0.3) + (std_latency_score * 0.2)
        return total
    
    # Sort results
    sorted_results = sorted(results, key=telephony_score, reverse=True)
    
    # Print top 3 configurations
    for i, result in enumerate(sorted_results[:3]):
        if result.get('status') != 'success':
            continue
            
        print(f"\n{i+1}. Model: {result.get('model_name')}, Voice: {result.get('voice_name')}")
        print(f"   Model ID: {result.get('model_id')}")
        print(f"   Voice ID: {result.get('voice_id')}")
        print(f"   Streaming Latency: {result.get('streaming_latency', 0):.3f}s")
        print(f"   Standard Latency: {result.get('standard_latency', 0):.3f}s")
        
        freq_metrics = result.get('frequency_metrics', {})
        if 'telephony_ratio' in freq_metrics:
            print(f"   Telephony Frequency Ratio: {freq_metrics.get('telephony_ratio', 0):.3f}")
        if 'clarity_score' in freq_metrics:
            print(f"   Speech Clarity Score: {freq_metrics.get('clarity_score', 0):.3f}")
        if 'duration' in freq_metrics:
            print(f"   Audio Duration: {freq_metrics.get('duration', 0):.3f}s")
        
        score = telephony_score(result)
        print(f"   Overall Telephony Score: {score:.2f}/10")
    
    # Final recommendation
    if sorted_results and sorted_results[0].get('status') == 'success':
        best = sorted_results[0]
        print("\n=== RECOMMENDED CONFIGURATION ===")
        print(f"Model ID: {best.get('model_id')}")
        print(f"Model Name: {best.get('model_name')}")
        print(f"Voice ID: {best.get('voice_id')}")
        print(f"Voice Name: {best.get('voice_name')}")
        print("\nAdd these to your configuration when implementing ElevenLabs TTS integration.")
    else:
        print("\nNo successful tests completed. Please check your API key and try again.")

if __name__ == "__main__":
    asyncio.run(main())