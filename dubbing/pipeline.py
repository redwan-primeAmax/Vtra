import sys
import logging
from dubbing.config import DubbingConfig
from dubbing.media import validate_input_file, extract_audio_and_bgm
from dubbing.transcribe import transcribe_audio
from dubbing.punctuation import restore_punctuation_and_sentences
from dubbing.translate import translate_sentences_google
from dubbing.tts import synthesize_speech
from dubbing.mixer import sync_and_assemble_video
from dubbing.memory import flush_memory

logger = logging.getLogger("dubbing.pipeline")

def print_step(msg: str):
    """টার্মিনালে রিয়েলটাইমে তাৎক্ষণিক স্টেটাস দেখানোর জন্য লগার ফাংশন।"""
    print(f"\n[PIPELINE] {msg}", flush=True)

class DubbingPipeline:
    def __init__(self, config: DubbingConfig):
        self.config = config

    def run(self):
        print_step("১/৭: ইনপুট ভ্যালিডেশন...")
        validate_input_file(self.config.input_video)
        flush_memory()

        print_step("২/৭: BGM সেপারেশন (Demucs)...")
        raw_wav, bgm_wav = extract_audio_and_bgm(self.config.input_video, self.config.work_dir)
        flush_memory()

        print_step("৩/৭: ট্রান্সক্রিপশন (Whisper Large-v3)...")
        raw_segments = transcribe_audio(raw_wav, model_size=self.config.whisper_model, compute_type=self.config.compute_type)
        flush_memory()

        print_step("৪/৭: সেগমেন্টেশন ও বাক্য বিন্যাস...")
        sentences = restore_punctuation_and_sentences(raw_segments, max_duration=self.config.max_segment_duration)
        flush_memory()

        print_step("৫/৭: স্মার্ট গুগল ট্রান্সলেটর দিয়ে বাংলা অনুবাদ...")
        translated = translate_sentences_google(sentences, glossary=self.config.glossary)
        flush_memory()

        print_step("৬/৭: প্রাকৃতিক বাংলা ভয়েস ওভার (Edge-TTS)...")
        tts_dir = self.config.work_dir / "tts_clips"
        synthesized = synthesize_speech(translated, tts_dir, voice=self.config.tts_voice)
        flush_memory()

        print_step("৭/৭: ফাইনাল মিক্সিং ও ভিডিও সিঙ্কিং...")
        sync_and_assemble_video(
            video_path=self.config.input_video,
            bgm_wav=bgm_wav,
            items=synthesized,
            output_dir=self.config.segments_dir,
            output_video=self.config.output_video,
            bgm_volume=self.config.bgm_volume
        )
        flush_memory()

        print_step(f"✅ ডাবিং সফলভাবে সম্পন্ন হয়েছে: {self.config.output_video.name}")
