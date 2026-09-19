from dataclasses import dataclass
from pathlib import Path

@dataclass
class DubbingConfig:
    input_video: Path
    output_video: Path
    work_dir: Path = Path("./dubbing_workspace")
    device: str = "cuda"
    compute_type: str = "float16"  # T4 VRAM-এর জন্য উপযুক্ত
    whisper_model: str = "large-v3"
    tts_model: str = "facebook/mms-tts-ben"
    translation_model: str = "facebook/nllb-200-distilled-600M"
    source_lang: str = "eng_Latn"
    target_lang: str = "ben_Beng"

    def __post_init__(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.work_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.work_dir / "pipeline_state.json"
