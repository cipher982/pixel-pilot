# pixel-pilot
AI agent for completing computer tasks through a dual-path architecture.

## Project Overview
Pixel Pilot is an AI-driven system that automates interactions on your computer through a sophisticated dual-path architecture, handling both terminal-based and visual-based workflows. The system can intelligently switch between these modes based on the task requirements.

## Key Components

1. Dual-Path Architecture:
   - Terminal Path: Executes command-line operations and processes text-based interactions
   - Visual Path: Handles GUI interactions through screenshot analysis and mouse/keyboard actions
   - Smart path switching based on task context

2. Core Systems:
   - Graph-based Workflow Engine: Manages task execution flow and state transitions
   - State Management: Unified state tracking across terminal and visual operations
   - LLM Integration: Supports multiple providers (OpenAI, local models, Bedrock, Fireworks)
   - Visual Operations: Screenshot capture and analysis for GUI interaction
   - Terminal Operations: Command execution and output processing

3. Tools and Utilities:
   - Mouse and Keyboard Control
   - Window Capture System
   - Audio Capture and Transcription (optional)
   - Terminal Command Execution
   - Visual Element Detection

## Getting Started

1. Requirements:
   - Python >= 3.12
   - UV package manager (`pip install uv`)

2. Setup:
   ```bash
   uv sync
   ```

3. Configuration:
   - Create a `.env` file with required API keys (e.g., OPENAI_API_KEY)
   - Optional: Configure task profiles in YAML format

4. Basic Usage:
   ```bash
   # Run the agent with a task profile
   uv run python pixelpilot/main.py --task-profile path/to/task.yaml

   # Run with a simple command
   uv run python -m pixelpilot.main -i "Open Notepad and type 'Hello, World!'"
   ```

5. Command Line Options:
   - `--enable-audio`: Enable audio capture and processing
   - `--debug`: Run in debug mode with test images
   - `--task-profile`: Path to task configuration YAML
   - `--instructions`: Direct task instructions override
   - `--llm-provider`: Choose LLM provider (openai/local/bedrock/fireworks)
   - `--label-boxes`: Enable visual debugging of detected elements

## Task Profiles
Tasks can be defined in YAML profiles containing:
- Task instructions
- Specific configurations for terminal/visual operations
- Custom workflow parameters

The system will automatically determine whether to start in terminal or visual mode based on the task requirements.

