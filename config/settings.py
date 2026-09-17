import os
import torch

class Config:
    # GPU Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Directory Settings
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(BASE_DIR, "inputs")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    TEMP_DIR = os.path.join(BASE_DIR, "temp")

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

# ডিরেক্টরিগুলো স্বয়ংক্রিয়ভাবে তৈরি হবে
os.makedirs(Config.INPUT_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.TEMP_DIR, exist_ok=True)

# সাধারণ ব্যবহারের জন্য ভ্যারিয়েবল সাপোর্ট
DEVICE = Config.DEVICE
WHISPER_MODEL_SIZE = Config.WHISPER_MODEL_SIZE
PUNCTUATION_MODEL_NAME = Config.PUNCTUATION_MODEL_NAME
TRANSLATION_MODEL_NAME = Config.TRANSLATION_MODEL_NAME
TTS_VOICE = Config.TTS_VOICE
