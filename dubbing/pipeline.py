import logging
from dubbing.config import DubbingConfig
from dubbing.media import validate_input_file, extract_audio
from dubbing.transcribe import transcribe_audio
from dubbing.punctuation import restore_punctuation_and_sentences
from dubbing.translate import translate_sentences_api
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

        logger.info("২/৭: অডিও এক্সট্র্যাকশন...")
        raw_wav = self.config.work_dir / "extracted_16k.wav"
        extract_audio(self.config.input_video, raw_wav)
        flush_memory()

        logger.info("৩/৭: ট্রান্সক্রিপশন (Whisper)...")
        raw_segments = transcribe_audio(
            raw_wav, 
            model_size=self.config.whisper_model, 
            compute_type=self.config.compute_type
        )
        flush_memory()

        logger.info("৪/৭: পাংচুয়েশন ও সেগমেন্টেশন (সর্বোচ্চ ৫ সেকেন্ডের ব্লক)...")
        sentences = restore_punctuation_and_sentences(raw_segments, max_duration=self.config.max_segment_duration)
        flush_memory()

        logger.info("৫/৭: API ভিত্তিক বাংলা অনুবাদ...")
        translated = translate_sentences_api(sentences, api_key=self.config.api_key)
        flush_memory()

        logger.info("৬/৭: বাংলা TTS সিন্থেসিস...")
        tts_dir = self.config.work_dir / "tts_clips"
        synthesized = synthesize_speech(translated, tts_dir, model_name=self.config.tts_model)
        flush_memory()

        logger.info("৭/৭: অডিও-ভিডিও স্পিড সিঙ্ক এবং ভিডিও অ্যাসেম্বলি...")
        sync_and_assemble_video(
            video_path=self.config.input_video,
            items=synthesized,
            output_dir=self.config.segments_dir,
            output_video=self.config.output_video
        )
        flush_memory()

        logger.info(f"ডাবিং এবং স্পিড-সিঙ্কিং সফলভাবে সম্পন্ন হয়েছে: {self.config.output_video}")
