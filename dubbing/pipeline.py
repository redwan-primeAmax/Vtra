import logging
from dubbing.config import DubbingConfig
from dubbing.media import validate_input_file, extract_audio_and_bgm
from dubbing.tts import synthesize_speech
from dubbing.mixer import sync_and_assemble_video
from dubbing.memory import flush_memory

logger = logging.getLogger("dubbing.pipeline")


def print_step(msg: str):
    print(f"\n[PIPELINE] {msg}", flush=True)


class DubbingPipeline:
    def __init__(self, config: DubbingConfig, pre_translated_segments=None):
        """
        pre_translated_segments: List[Dict] | None
            List হলে → শুধু TTS + Mixer (স্বাভাবিক ব্যবহার)
            None হলে → ত্রুটি (এই ভার্সনে অটো-অনুবাদ বাদ দেওয়া হয়েছে)
        """
        self.config = config
        self.pre_translated_segments = pre_translated_segments

    def run(self):
        if self.pre_translated_segments is None:
            raise ValueError(
                "pre_translated_segments প্রয়োজন। "
                "অটো-অনুবাদ বাদ দেওয়া হয়েছে — translated.json দিন।"
            )

        print_step("১: ইনপুট ভ্যালিডেশন...")
        validate_input_file(self.config.input_video)
        flush_memory()

        print_step("২: BGM সেপারেশন (Demucs)...")
        raw_wav, bgm_wav = extract_audio_and_bgm(
            self.config.input_video, self.config.work_dir
        )
        flush_memory()

        print_step("৩: Pre-translated script লোড...")
        translated = [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "src_text": "",
                "tgt_text": s["text"],
            }
            for s in self.pre_translated_segments
        ]
        print(f"   মোট {len(translated)} সেগমেন্ট")

        print_step("৪: প্রাকৃতিক বাংলা ভয়েস ওভার (Edge-TTS)...")
        tts_dir = self.config.work_dir / "tts_clips"
        synthesized = synthesize_speech(
            translated, tts_dir, voice=self.config.tts_voice
        )
        flush_memory()

        print_step("৫: ফাইনাল মিক্সিং ও ভিডিও সিঙ্কিং...")
        sync_and_assemble_video(
            video_path=self.config.input_video,
            bgm_wav=bgm_wav,
            items=synthesized,
            output_dir=self.config.segments_dir,
            output_video=self.config.output_video,
            bgm_volume=self.config.bgm_volume,
        )
        flush_memory()

        print_step(f"✅ ডাবিং সফলভাবে সম্পন্ন হয়েছে: {self.config.output_video.name}")