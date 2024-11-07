import io
import os
import queue
import time
import wave
from typing import Optional
from dotenv import load_dotenv

import numpy as np
import openai
import sounddevice as sd

load_dotenv()

class AudioCapture:
	def __init__(self, device_name: str):
		self.device_name = device_name
		self.audio_queue = queue.Queue()
		self.is_running = False
		self.stream = None
		# Set API key from environment variable
		openai.api_key = os.getenv('OPENAI_API_KEY')
		if not openai.api_key:
			raise ValueError("OPENAI_API_KEY environment variable is not set")
		
	def audio_callback(self, indata: np.ndarray, frames: int, 
					  time_info: dict, status: sd.CallbackFlags) -> None:
		"""Callback function for audio stream."""
		if status:
			print(f"Status: {status}")
		self.audio_queue.put(np.copy(indata))
	
	def start_capture(self) -> None:
		"""Start capturing audio."""
		self.is_running = True
		self.stream = sd.InputStream(
			callback=self.audio_callback,
			channels=2,
			samplerate=44100,
			device=self.device_name
		)
		self.stream.start()
	
	def stop_capture(self) -> None:
		"""Stop capturing audio."""
		self.is_running = False
		if self.stream:
			self.stream.stop()
			self.stream.close()
	
	def _prepare_audio_file(self, audio_data: np.ndarray) -> io.BytesIO:
		"""Convert audio data to WAV format suitable for OpenAI API."""
		byte_io = io.BytesIO()
		with wave.open(byte_io, 'wb') as wav_file:
			wav_file.setnchannels(2)
			wav_file.setsampwidth(2)  # 16-bit
			wav_file.setframerate(44100)
			wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
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
			# Send to OpenAI Whisper API
			response = openai.Audio.transcribe(
				"whisper-1",
				audio_file,
				response_format="text"
			)
			return response
		except Exception as e:
			print(f"Error during transcription: {e}")
			return None

if __name__ == "__main__":
	import time
	
	# Simple test
	capture = AudioCapture()

	print("Starting audio capture...")
	capture.start_capture()
	
	try:
		for _ in range(5):  # Test for 5 seconds
			time.sleep(1)
			text = capture.get_transcription()
			if text:
				print(f"Transcribed: {text}")
	finally:
		capture.stop_capture()