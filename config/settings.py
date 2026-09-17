import os
import torch

# GPU Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory Settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Models Configuration
WHISPER_MODEL_SIZE = "large-v3-turbo"
PUNCTUATION_MODEL_NAME = "oliverguhr/fullstop-punctuation-multilingual-sonar"
TRANSLATION_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# TTS Settings
TTS_VOICE = "bn-BD-NabanitaNeural"

# Video Processing Settings
TARGET_FPS = 30
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
