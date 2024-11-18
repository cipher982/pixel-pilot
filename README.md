# auto-comply
AI agent for completing computer tasks.

## Project Overview
This project is designed to automate interactions on your computer or through websites by capturing audio, video, and images, and then performing actions such as clicking through the content. The system leverages audio transcription and image capture to make informed decisions about the next steps in the course or quiz.

Key Components
1. Audio Capture and Transcription:
   - Utilizes a virtual audio device to capture audio from the system.
   - Transcribes the captured audio using OpenAI's Whisper API to understand spoken commands or content.
2. Window Capture:
   - Captures screenshots of the specified window.
   - Allows interactive selection of the window to be controlled.
3. Action System:
   - Decides on actions based on transcribed audio and captured screenshots.
   - Performs actions such as clicking on specific areas of the screen to navigate through content.
4. Configuration:
   - Provides configurable intervals for audio processing, screenshot capturing, and action execution to optimize performance and responsiveness.
5. Main Agent:
   - Integrates all components into a cohesive system.
   - Continuously captures audio and screenshots, processes them, and executes actions in a loop.

