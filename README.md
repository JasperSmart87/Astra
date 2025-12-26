# Astra LM Studio Sensory Bridge

This package provides an example implementation of a multimodal agent that connects your webcam and microphone to a local LM Studio model.  It captures camera frames and microphone audio, generates short summaries (or captions via a vision‑language model), decides whether to speak or remain silent, and then forwards the sensory updates to your chosen LM Studio chat model.

## Prerequisites

* Python 3.8 or later
* A recent version of [LM Studio](https://lmstudio.ai) running on your machine with the local server enabled.
* The required Python packages listed in `requirements.txt` (install with `pip install -r requirements.txt`).

## Installation

1. Clone or extract this repository.
2. Open a terminal in the project directory.
3. (Optional) Create and activate a Python virtual environment.
4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Copy `.env.example` to `.env` and set the following environment variables:

* `LMSTUDIO_BASE_URL` – The base URL of your LM Studio local server, e.g. `http://localhost:1234/v1`.
* `LMSTUDIO_MODEL` – The ID of the chat model you want to use (from LM Studio’s model list).
* `VLM_MODEL` (optional) – The ID of the vision‑language model if you want image captions.
* `WAKE_NAME` – The name Astra will respond to (default is `Astra`).

You can also adjust other parameters in `config.py`, such as camera index, sample rate, and injection frequency.

## Running

Make sure LM Studio’s local server is running, then execute:

```bash
python orchestrator.py
```

The agent will start capturing sensory input and print logs to the console.  It will only speak via your chat model when it hears its wake name or is otherwise prompted.

Press `Ctrl+C` to stop the program.

## Contents

- `orchestrator.py` – Main loop that ties everything together.
- `presence_engine.py` – Controls when Astra should respond.
- `vision_agent.py` – Optional module for calling a vision‑language model.
- `lmstudio_client.py` – Simple HTTP client for LM Studio’s API.
- `config.py` – Central configuration values.
- `perception/` – Modules for audio and camera capture and speech‑to‑text.
- `memory/` – Ring buffer utility for recent events.
- `agent/` – Prompt templates and system messages.