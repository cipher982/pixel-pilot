import io
import os
import queue
import time
import wave
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class AudioCapture:
    """Captures audio from virtual audio device and transcribes using Whisper API."""

    def __init__(self, device_name: str = "BlackHole 2ch"):
        """Initialize audio capture with specified device.

        Args:
            device_name: Name of the virtual audio device (default: BlackHole 2ch)
        """
        # OpenAI setup
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Create client instance
        if not self.client.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Audio device setup
        self.devices = sd.query_devices()
        self.device_id = self._find_device(device_name)
        if self.device_id is None:
            raise ValueError(
                f"Could not find {device_name}! " "Please ensure it's installed: brew install blackhole-2ch"
            )

        # Verify it's not a microphone device
        device_info = self.devices[self.device_id]
        if device_info.get("hostapi") == 0:  # CoreAudio on macOS
            if "input" in device_info.get("name", "").lower():
                raise ValueError(f"{device_name} appears to be a microphone input device")

        # Audio settings (optimized for Whisper)
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.channels = 1  # Mono audio
        self.chunk_duration = 30  # Whisper processes 30-second chunks
        self.samples_per_chunk = self.sample_rate * self.chunk_duration

        # Stream state
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.stream = None

    @staticmethod
    def list_available_devices() -> List[Dict]:
        """List all available audio input devices."""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append(
                    {
                        "id": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "sample_rate": device["default_samplerate"],
                    }
                )
        return devices

    def _find_device(self, device_name: str) -> Optional[int]:
        """Find device ID by name."""
        for i, device in enumerate(self.devices):
            if device["max_input_channels"] > 0 and device_name.lower() in device["name"].lower():
                return i
        return None

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Callback function for audio stream."""
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(np.copy(indata))

    def start_capture(self) -> None:
        """Start capturing audio with Whisper-compatible settings."""
        self.is_running = True
        self.stream = sd.InputStream(
            callback=self.audio_callback, channels=self.channels, samplerate=self.sample_rate, device=self.device_id
        )
        self.stream.start()

    def stop_capture(self) -> None:
        """Stop capturing audio."""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def _prepare_audio_file(self, audio_data: np.ndarray) -> io.BytesIO:
        """Convert audio data to WAV format matching Whisper's requirements."""
        # Ensure mono audio (average if stereo)
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)

        # Normalize to [-1, 1] range
        audio_data = audio_data.flatten()
        audio_data = np.clip(audio_data, -1.0, 1.0)

        byte_io = io.BytesIO()
        with wave.open(byte_io, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        # Add .wav extension to the BytesIO object
        byte_io.name = "audio.wav"
        byte_io.seek(0)
        return byte_io

    def get_transcription(self) -> Optional[str]:
        """Get transcription of accumulated audio using OpenAI's Whisper API."""
        if self.audio_queue.empty():
            return None

        # Collect all available audio data
        audio_chunks = []
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                audio_chunks.append(chunk)
            except queue.Empty:
                break

        if not audio_chunks:
            return None

        # Combine chunks and prepare audio file
        audio_data = np.concatenate(audio_chunks)
        audio_file = self._prepare_audio_file(audio_data)

        try:
            with audio_file as af:
                response = self.client.audio.transcriptions.create(model="whisper-1", file=af)
            return response.text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None


if __name__ == "__main__":
    # Print available devices
    print("Available input devices:")
    for device in AudioCapture.list_available_devices():
        print(f"ID: {device['id']}, Name: {device['name']}")

    # Simple test
    try:
        capture = AudioCapture()
        print(f"Capturing audio from: {capture.devices[capture.device_id]['name']}")

        print("Starting audio capture...")
        capture.start_capture()

        print("Listening for 5 seconds...")
        for _ in range(5):  # Test for 5 seconds
            time.sleep(1)
            text = capture.get_transcription()
            if text:
                print(f"Transcribed: {text}")

    except KeyboardInterrupt:
        print("\nStopping capture...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if "capture" in locals():
            capture.stop_capture()
