"""Configuration values for the Astra LM Studio sensory bridge.

Most values can be overridden via environment variables.  See `.env.example` for the variables used by this module.
"""

from dataclasses import dataclass
import os


@dataclass
class Settings:
    """Container for runtime configuration parameters."""

    # Camera settings
    camera_index: int = 0
    camera_fps: int = 1
    camera_downscale_width: int = 320

    # Audio settings
    audio_sample_rate: int = 44100
    audio_chunk_seconds: float = 0.25
    audio_device: str | None = None

    # Event injection
    inject_every_seconds: int = 6
    max_context_events: int = 10

    # LM Studio connection
    lmstudio_base_url: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    lmstudio_api_key: str | None = os.getenv("LMSTUDIO_API_KEY", None)
    lmstudio_model: str = os.getenv("LMSTUDIO_MODEL", "")

    # Wake name (what the agent listens for)
    wake_name: str = os.getenv("WAKE_NAME", "Astra")


# Export a default instance for convenience
SETTINGS = Settings()