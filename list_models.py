from __future__ import annotations
from config import SETTINGS
from lmstudio_client import LMStudioClient

if __name__ == "__main__":
    c = LMStudioClient(SETTINGS.lmstudio_base_url, SETTINGS.lmstudio_api_key)
    data = c.list_models()
    print(data)
