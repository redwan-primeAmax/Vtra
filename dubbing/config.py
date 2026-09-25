from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dubbing.glossary import load_glossary


@dataclass
class DubbingConfig:
    input_video: Path
    output_video: Path
    work_dir: Path = Path("./dubbing_workspace")
    device: str = "cuda"
    compute_type: str = "float16"
    whisper_model: str = "large-v3"
    tts_voice: str = "bn-BD-NabanitaNeural"
    max_segment_duration: float = 5.0
    bgm_volume: float = 0.3

    # glossary/ ফোল্ডারের সব .txt থেকে অটো-লোড হবে (dedup, order preserved)
    glossary: List[str] = field(default_factory=load_glossary)

    def __post_init__(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.work_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        self.segments_dir = self.work_dir / "retimed_segments"
        self.segments_dir.mkdir(exist_ok=True)