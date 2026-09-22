from dataclasses import dataclass
from pathlib import Path

@dataclass
class DubbingConfig:
    input_video: Path
    output_video: Path
    api_key: str = ""  # Gemini/OpenAI API Key
    work_dir: Path = Path("./dubbing_workspace")
    device: str = "cuda"
    compute_type: str = "float16"
    whisper_model: str = "large-v3"
    tts_model: str = "facebook/mms-tts-ben"
    max_segment_duration: float = 5.0  # পাংচুয়েশন না থাকলে সর্বোচ্চ সময় (সেকেন্ড)

    def __post_init__(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.work_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        self.segments_dir = self.work_dir / "retimed_segments"
        self.segments_dir.mkdir(exist_ok=True)
