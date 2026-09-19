import logging
from dubbing.config import DubbingConfig
from dubbing.media import validate_input_file, extract_audio
from dubbing.transcribe import transcribe_audio
from dubbing.punctuation import restore_punctuation_and_sentences
from dubbing.translate import translate_sentences
from dubbing.timing import adapt_timing_budget
from dubbing.tts import synthesize_speech
from dubbing.mixer import time_fit_and_mix, assemble_final_video

logger = logging.getLogger("dubbing.pipeline")

class DubbingPipeline:
    def __init__(self, config: DubbingConfig):
        self.config = config

    def run(self):
        logger.info("১/৮: ভ্যালিডেশন শুরু হচ্ছে...")
        media_info = validate_input_file(self.config.input_video)
        duration = float(media_info["format"]["duration"])

        logger.info("২/৮: অডিও এক্সট্র্যাকশন...")
        raw_wav = self.config.work_dir / "extracted_16k.wav"
        extract_audio(self.config.input_video, raw_wav)

        logger.info("৩/৮: ট্রান্সক্রিপশন (Whisper Large-v3)...")
        raw_segments = transcribe_audio(
            raw_wav, 
            model_size=self.config.whisper_model, 
            compute_type=self.config.compute_type
        )

        logger.info("৪/৮: সেগমেন্টেশন ও বাক্য বিন্যাস...")
        sentences = restore_punctuation_and_sentences(raw_segments)

        logger.info("৫/৮: অনুবাদ (ইংরেজি → বাংলা)...")
        translated = translate_sentences(sentences, model_name=self.config.translation_model)

        logger.info("৬/৮: টাইমিং বাজেট অ্যাডাপ্টেশন...")
        timed_items = adapt_timing_budget(translated)

        logger.info("৭/৮: বাংলা TTS সিন্থেসিস...")
        tts_dir = self.config.work_dir / "tts_clips"
        synthesized = synthesize_speech(timed_items, tts_dir, model_name=self.config.tts_model)

        logger.info("৮/৮: মিক্সিং এবং ভিডিও অ্যাসেম্বলি...")
        final_wav = self.config.work_dir / "final_bn.wav"
        time_fit_and_mix(synthesized, duration, final_wav)
        assemble_final_video(self.config.input_video, final_wav, self.config.output_video)

        logger.info(f"ডাবিং সফলভাবে শেষ হয়েছে: {self.config.output_video}")
