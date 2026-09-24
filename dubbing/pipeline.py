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

class DubbingPipeline:
    def __init__(self, config: DubbingConfig):
        self.config = config

    def run(self):
        logger.info("১/৭: ইনপুট ভ্যালিডেশন...")
        validate_input_file(self.config.input_video)
        flush_memory()

        logger.info("২/৭: BGM সেপারেশন (Demucs)...")
        raw_wav, bgm_wav = extract_audio_and_bgm(self.config.input_video, self.config.work_dir)
        flush_memory()

        logger.info("৩/৭: ট্রান্সক্রিপশন (Whisper Large-v3)...")
        raw_segments = transcribe_audio(raw_wav, model_size=self.config.whisper_model, compute_type=self.config.compute_type)
        flush_memory()

        logger.info("৪/৭: সেগমেন্টেশন...")
        sentences = restore_punctuation_and_sentences(raw_segments, max_duration=self.config.max_segment_duration)
        flush_memory()

        logger.info("৫/৭: গুগল ট্রান্সলেট ফিল্টার্ড বাংলা অনুবাদ...")
        translated = translate_sentences_google(sentences, glossary=self.config.glossary)
        flush_memory()

        logger.info("৬/৭: প্রাকৃতিক বাংলা ভয়েস ওভার (Edge-TTS)...")
        tts_dir = self.config.work_dir / "tts_clips"
        synthesized = synthesize_speech(translated, tts_dir, voice=self.config.tts_voice)
        flush_memory()

        logger.info("৭/৭: ফাইনাল মিক্সিং ও সিঙ্কিং...")
        sync_and_assemble_video(
            video_path=self.config.input_video,
            bgm_wav=bgm_wav,
            items=synthesized,
            output_dir=self.config.segments_dir,
            output_video=self.config.output_video,
            bgm_volume=self.config.bgm_volume
        )
        flush_memory()

        logger.info(f"ডাবিং সফলভাবে সম্পন্ন হয়েছে: {self.config.output_video}")
