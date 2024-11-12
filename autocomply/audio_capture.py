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

from autocomply.logger import setup_logger

logger = setup_logger(__name__)

load_dotenv()


AUDIO_INTERVAL = 5


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
        self.device_id = self._find_system_audio_device()
        if self.device_id is None:
            self.device_id = self._find_device(device_name)
            if self.device_id is None:
                raise ValueError(
                    "Could not find suitable audio capture device. "
                    "On macOS 12+, enable 'Audio Capture' in Security & Privacy settings. "
                    "Otherwise, install BlackHole: brew install blackhole-2ch"
                )

        # Verify device type
        device_info = self.devices[self.device_id]
        logger.info(f"Selected audio device: {device_info['name']}")
        if device_info.get("hostapi") == 0 and "input" in device_info.get("name", "").lower():
            raise ValueError(f"{device_name} appears to be a microphone input device")

        # Audio processing settings
        # Audio processing settings
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.channels = 1  # Mono audio
        self.chunk_duration = 30  # Restore original chunk duration
        self.min_audio_length = 3 * self.sample_rate  # Min 3 seconds
        self.energy_threshold = 0.001  # Restore original threshold

        # Stream state management
        # Stream state management
        self.stream = None
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.streaming_buffer = []
        self.last_transcription = 0
        self.audio_process_interval = AUDIO_INTERVAL

        # Visualization settings
        self.console = Console()
        self.blocks = " ▁▂▃▄▅▆▇█"  # Volume visualization characters
        self.current_chunk_viz = []
        self.current_chunk_start = time.time()  # Restore this for visualization

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

    def _find_system_audio_device(self) -> Optional[int]:
        """Find macOS system audio capture device."""
        logger.info("Searching for system audio capture device...")
        for i, device in enumerate(self.devices):
            name = device["name"].lower()
            max_input_channels = device["max_input_channels"]
            logger.info(f"Device: {device['name']} (ID: {i}) has {max_input_channels} input channels")
            logger.info(f"Device full info: {device}")

            # Look for BlackHole with input capability
            if max_input_channels > 0 and "blackhole" in name:
                logger.info(f"Found BlackHole device: {device['name']} (ID: {i})")
                return i

        logger.warning("BlackHole audio capture device not found.")
        return None

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

        # Generate visualization and store it
        samples = np.abs(indata[:, 0])
        samples = samples[:: len(samples) // 32]
        peak = np.max(samples)
        normalized = samples / max(peak, 0.01)

        viz_data = {"normalized": normalized, "peak": peak, "timestamp": time.time()}
        self.current_chunk_viz.append(viz_data)

        self.audio_queue.put(np.copy(indata))

    def _render_visualization(self, viz_data_list) -> None:
        """Render a single compressed visualization for the entire chunk."""
        if not viz_data_list:
            return

        # Combine all normalized data and downsample to ~32 points
        all_samples = np.concatenate([vd["normalized"] for vd in viz_data_list])
        chunk_size = len(all_samples) // 32
        if chunk_size > 0:
            # Use mean instead of max for more variation
            compressed = [np.mean(all_samples[i : i + chunk_size]) for i in range(0, len(all_samples), chunk_size)]
        else:
            compressed = all_samples

        # Normalize the compressed data
        max_val = np.max(compressed)
        min_val = np.min(compressed)
        if max_val > min_val:
            compressed = (compressed - min_val) / (max_val - min_val)

        # Create single visualization
        chars = [self.blocks[int(n * (len(self.blocks) - 1))] for n in compressed]
        colored_chars = []
        for n, char in zip(compressed, chars):
            if n > 0.8:
                color = "bright_red"
            elif n > 0.5:
                color = "yellow"
            else:
                color = "green"
            colored_chars.append(f"[{color}]{char}[/]")

        waveform = "".join(colored_chars)
        peak = max(vd["peak"] for vd in viz_data_list)
        self.console.print(f"♪ [{waveform}] {peak:.3f}")

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
        """Get transcription of accumulated audio using OpenAI's Whisper API."""
        current_time = time.time()
        if current_time - self.last_transcription < self.audio_process_interval or self.audio_queue.empty():
            return None

        # Collect audio chunks
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

        if not audio_chunks or total_samples < self.min_audio_length:
            return None

        try:
            # Combine audio chunks
            audio_data = np.concatenate(audio_chunks)

            # Check energy levels
            chunk_size = int(0.1 * self.sample_rate)  # 100ms windows
            energy = np.array(
                [np.sqrt(np.mean(chunk**2)) for chunk in np.array_split(audio_data, len(audio_data) // chunk_size)]
            )

            if np.max(energy) < self.energy_threshold:
                return None

            # Prepare and transcribe
            audio_file = self._prepare_audio_file(audio_data)
            response = self.client.audio.transcriptions.create(model="whisper-1", file=audio_file, language="en")

            # Print visualizations and transcription
            logger.info("=== Audio Chunk ===")
            self._render_visualization(self.current_chunk_viz)
            logger.info(f"Transcription: {response.text}")
            logger.info("================")

            # Reset state
            self.current_chunk_viz = []
            self.current_chunk_start = time.time()
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

        logger.info(f"Listening (checking every {AUDIO_INTERVAL} seconds)...")
        while True:
            time.sleep(AUDIO_INTERVAL)
            text = capture.get_transcription()

    except KeyboardInterrupt:
        logger.info("Stopping capture...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if "capture" in locals():
            capture.stop_capture()
