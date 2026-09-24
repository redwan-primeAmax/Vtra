from dataclasses import dataclass, field
from pathlib import Path
from typing import List

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
    # যেসব শব্দ অনুবাদ হবে না তার তালিকা
    glossary: List[str] = field(default_factory=lambda: [
        "Ballon d'Or", "Champions League", "Real Madrid", "Barcelona", "Messi", "Ronaldo"
    ])

    def __post_init__(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.work_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        self.segments_dir = self.work_dir / "retimed_segments"
        self.segments_dir.mkdir(exist_ok=True)
