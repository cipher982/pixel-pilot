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
from rich.console import Console

from autocomply.config import Config
from autocomply.logger import setup_logger

logger = setup_logger(__name__)

load_dotenv()


class AudioCapture:
    """Captures audio from virtual audio device and transcribes using Whisper API."""

    def __init__(self, device_name: str = "BlackHole 2ch"):
        """Initialize audio capture with specified device.

        Args:
            device_name: Name of the virtual audio device (default: BlackHole 2ch)
        """
        # OpenAI setup
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Audio device setup
        self.devices = sd.query_devices()
        self.device_id = self._find_device(device_name)
        if self.device_id is None:
            raise ValueError(f"Could not find {device_name}! Please ensure it's installed: brew install blackhole-2ch")

        # Verify it's not a microphone device
        device_info = self.devices[self.device_id]
        logger.info(f"Selected audio device: {device_info['name']}")
        logger.info(f"Device details: {device_info}")
        if device_info.get("hostapi") == 0:  # CoreAudio on macOS
            if "input" in device_info.get("name", "").lower():
                raise ValueError(f"{device_name} appears to be a microphone input device")

        # Audio settings (optimized for Whisper)
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.channels = 1  # Mono audio
        self.chunk_duration = 30  # Whisper processes 30-second chunks
        self.samples_per_chunk = self.sample_rate * self.chunk_duration
        self.min_audio_length = 3 * self.sample_rate  # Min 3 seconds before transcribing

        # Stream state
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.stream = None

        # Visualization settings
        self.console = Console()
        self.blocks = " ▁▂▃▄▅▆▇█"  # Granular volume visualization
        self.last_visualization = 0
        self.visualization_interval = Config.MAIN_LOOP_INTERVAL

        # Timing controls
        self.last_transcription = 0
        self.audio_process_interval = Config.AUDIO_INTERVAL

    @staticmethod
    def list_available_devices() -> List[Dict]:
        """List all available audio input devices."""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            # Include all devices, not just input
            devices.append(
                {
                    "id": i,
                    "name": device["name"],
                    "channels": {"input": device["max_input_channels"], "output": device["max_output_channels"]},
                    "sample_rate": device["default_samplerate"],
                    "raw": device,  # Include all device info
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
            logger.warning(f"Status: {status}")

        # Only visualize at specified intervals
        current_time = time.time()
        if current_time - self.last_visualization >= self.visualization_interval:
            # Create detailed waveform visualization
            samples = np.abs(indata[:, 0])
            samples = samples[:: len(samples) // 32]

            peak = np.max(samples)
            normalized = samples / max(peak, 0.01)
            chars = [self.blocks[int(n * (len(self.blocks) - 1))] for n in normalized]

            colored_chars = []
            for n, char in zip(normalized, chars):
                if n > 0.8:
                    color = "bright_red"
                elif n > 0.5:
                    color = "yellow"
                else:
                    color = "green"
                colored_chars.append(f"[{color}]{char}[/]")

            waveform = "".join(colored_chars)
            self.console.print(f"♪ [{waveform}] {peak:.3f}")
            self.last_visualization = current_time

        self.audio_queue.put(np.copy(indata))

    def start_capture(self) -> None:
        """Start capturing audio with Whisper-compatible settings."""
        self.is_running = True
        try:
            device_info = sd.query_devices(self.device_id)
            logger.info(f"Starting capture with settings: {device_info['default_samplerate']}Hz")

            self.stream = sd.InputStream(  # Changed from Stream to InputStream
                callback=self.audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                device=self.device_id,  # Only specify input device
                dtype=np.float32,
            )
            self.stream.start()
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise

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
        """Get transcription of accumulated audio using OpenAI's Whisper API.

        Returns:
            Optional[str]: Transcribed text if successful, None otherwise
        """
        # Check timing interval
        current_time = time.time()
        if current_time - self.last_transcription < self.audio_process_interval:
            return None

        # Check if we have audio to process
        if self.audio_queue.empty():
            return None

        # Collect audio chunks up to our target duration
        audio_chunks = []
        total_samples = 0
        target_samples = self.sample_rate * self.chunk_duration

        while not self.audio_queue.empty() and total_samples < target_samples:
            try:
                chunk = self.audio_queue.get_nowait()
                total_samples += len(chunk)
                audio_chunks.append(chunk)
            except queue.Empty:
                break

        # Ensure minimum audio length
        if total_samples < self.min_audio_length:
            logger.info(f"Audio too short: {total_samples} samples < {self.min_audio_length} minimum")
            return None

        # Combine and process audio
        try:
            audio_data = np.concatenate(audio_chunks)
            logger.debug(
                f"Processing audio: shape={audio_data.shape}, duration={len(audio_data)/self.sample_rate:.1f}s"
            )

            # Prepare audio file for Whisper
            audio_file = self._prepare_audio_file(audio_data)

            # Send to Whisper API
            response = self.client.audio.transcriptions.create(model="whisper-1", file=audio_file, language="en")

            # Update last transcription time on success
            self.last_transcription = current_time

            return response.text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None


if __name__ == "__main__":
    # Print available devices
    logger.info("All available devices:")
    for device in AudioCapture.list_available_devices():
        logger.info(f"\nDevice ID: {device['id']}")
        logger.info(f"Name: {device['name']}")
        logger.info(f"Channels: {device['channels']}")
        logger.info(f"Raw info: {device['raw']}")

    # Simple test
    try:
        capture = AudioCapture()
        logger.info(f"Capturing audio from: {capture.devices[capture.device_id]['name']}")
        capture.start_capture()

        logger.info(f"Listening (checking every {Config.AUDIO_INTERVAL} seconds)...")
        while True:
            time.sleep(Config.MAIN_LOOP_INTERVAL)
            text = capture.get_transcription()
            if text:
                logger.info(f"Transcribed: {text}")

    except KeyboardInterrupt:
        logger.info("Stopping capture...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if "capture" in locals():
            capture.stop_capture()
