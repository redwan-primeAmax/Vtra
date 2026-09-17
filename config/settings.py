import torch

class Config:
    # ------------------------------------------------------------------
    # ১. হার্ডওয়্যার ও জিপিইউ কনফিগারেশন
    # ------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
    
    # ------------------------------------------------------------------
    # ২. এআই মডেল সেটিংস
    # ------------------------------------------------------------------
    WHISPER_MODEL = "large-v3-turbo"
    TRANSLATOR_MODEL = "facebook/nllb-200-distilled-600M"
    TTS_VOICE_BN = "bn-BD-NabanitaNeural"
    
    # ------------------------------------------------------------------
    # ৩. টাইম সিঙ্ক সেটিংস (০.৯৫x - ১.১০x লিমিট)
    # ------------------------------------------------------------------
    MIN_STRETCH_RATE = 0.95
    MAX_STRETCH_RATE = 1.10
    
    # ------------------------------------------------------------------
    # ৪. এক্সপোর্ট কনফিগারেশন
    # ------------------------------------------------------------------
    VIDEO_CODEC = "h264_nvenc" if torch.cuda.is_available() else "libx264"
    AUDIO_CODEC = "aac"
    AUDIO_BITRATE = "192k"
    PIXEL_FORMAT = "yuv420p"
    PRESET = "fast"
