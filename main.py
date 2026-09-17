import os
import shutil
import time
import threading
from utils.logger import logger
from utils.gpu import clear_vram
from config.settings import Config
from modules.input_handler import scan_and_prepare_input, extract_audio
from modules.audio_sep import separate_audio
from modules.stt_engine import transcribe_english, merge_short_segments, save_transcript_to_txt
from modules.translator import translate_en_to_bn, save_translation_to_txt
from modules.tts_engine import generate_bn_tts
from modules.time_sync import sync_and_merge_segments, get_audio_duration
from modules.exporter import merge_audio_tracks, render_final_video

def preload_models_in_background():
    try:
        logger.info(" [Background Worker] Qwen এবং Whisper মডেল কনফিগারেশন চেক শুরু হচ্ছে...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        AutoTokenizer.from_pretrained(Config.TRANSLATION_MODEL_NAME)
        AutoModelForCausalLM.from_pretrained(Config.TRANSLATION_MODEL_NAME)
        logger.info(" [Background Worker] মডেল প্রি-লোড সম্পন্ন হয়েছে!")
    except Exception as e:
        logger.warning(f" [Background Worker] ব্যাকগ্রাউন্ড মডেল লোডিং সতর্কবার্তা: {e}")

def main():
    start_time = time.time()
    logger.info("==================================================")
    logger.info("        AutoDubber Master Engine System           ")
    logger.info("==================================================")
    
    input_dir = Config.INPUT_DIR
    output_dir = Config.OUTPUT_DIR
    temp_dir = Config.TEMP_DIR
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Input Detection & Auto Conversion
        logger.info("[ধাপ ১/৭] ইনপুট ফাইল প্রসেসিং...")
        video_path = scan_and_prepare_input(input_dir, temp_dir)
        
        raw_audio = os.path.join(temp_dir, "extracted.wav")
        extract_audio(video_path, raw_audio)
        total_duration = get_audio_duration(raw_audio)
        logger.info(f"ভিডিও ডিউরেশন: {total_duration:.2f} সেকেন্ড")
        
        # ব্যাকগ্রাউন্ড মডেল প্রি-লোড
        preload_thread = threading.Thread(target=preload_models_in_background, daemon=True)
        preload_thread.start()
        
        # Step 2: Audio Separation (Demucs)
        logger.info("[ধাপ ২/৭] BGM এবং Vocal স্প্লিট করা হচ্ছে (Demucs)...")
        vocal_path, bgm_path = separate_audio(raw_audio, temp_dir)
        
        # Step 3: English Speech-to-Text (Whisper)
        logger.info("[ধাপ ৩/৭] ইংরেজি স্পিচ-টু-টেক্সট প্রসেসিং (faster-whisper)...")
        raw_segments = transcribe_english(vocal_path)
        logger.info(f"Whisper থেকে মোট {len(raw_segments)} টি কাঁচা সেগমেন্ট পাওয়া গেছে।")
        
        segments = merge_short_segments(raw_segments, min_duration=1.5)
        logger.info(f"স্মার্ট রি-গ্রুপিং এর পর মোট {len(segments)} টি বাক্য।")
        
        # 📄 ১. মূল ইংরেজি ট্রান্সক্রিপ্ট টেক্সট ফাইলে সেভ
        raw_txt_path = os.path.join(output_dir, "english_transcript.txt")
        save_transcript_to_txt(segments, raw_txt_path)
        
        # Step 4: Machine Translation (EN -> BN)
        logger.info("[ধাপ ৪/৭] ইংরেজি থেকে বাংলা অনুবাদ করা হচ্ছে (Qwen2.5)...")
        translated_segments = translate_en_to_bn(segments)
        logger.info(f"অনুবাদ সম্পন্ন: {len(translated_segments)} সেগমেন্ট")
        
        # 📄 ২. ইংরেজি + বাংলা অনুবাদ এনালাইসিস ফাইলে সেভ
        analysis_txt_path = os.path.join(output_dir, "translation_analysis.txt")
        save_translation_to_txt(translated_segments, analysis_txt_path)
        
        # Step 5: Bengali TTS Generation
        logger.info("[ধাপ ৫/৭] বাংলা ভয়েস ওভার জেনারেট করা হচ্ছে (Edge-TTS)...")
        tts_segments = generate_bn_tts(translated_segments, temp_dir, voice=Config.TTS_VOICE)
        logger.info(f"TTS জেনারেশন সম্পন্ন: {len(tts_segments)} অডিও ফাইল")
        
        # Step 6: Dynamic Time Stretching & Lip-Sync Alignment
        logger.info("[ধাপ ৬/৭] অডিও টাইম-স্ট্রেচিং ও সিঙ্কিং প্রসেস চালিত হচ্ছে...")
        synced_vocal = sync_and_merge_segments(
            tts_audio_paths=tts_segments,
            whisper_segments=translated_segments,
            output_dir=os.path.join(temp_dir, "synced"),
            output_file=os.path.join(temp_dir, "final_vocals.wav")
        )
        
        # Step 7: Final Audio-Video Merging
        logger.info("[ধাপ ৭/৭] চূড়ান্ত ভিডিও মার্জ ও ইউনিভার্সাল H.264 এনকোডিং...")
        mixed_audio = os.path.join(temp_dir, "final_mix.wav")
        output_video = os.path.join(output_dir, "final_dubbed_video.mp4")
        
        merge_audio_tracks(str(synced_vocal), bgm_path, mixed_audio)
        render_final_video(video_path, mixed_audio, output_video)
        
        # Temp Cleanup & VRAM Clear
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        clear_vram()
        
        elapsed = round(time.time() - start_time, 2)
        logger.info("==================================================")
        logger.info(f" ডাবিং সফলভাবে শেষ হয়েছে! মোট সময়: {elapsed}s")
        logger.info(f" ১. ফাইনাল ভিডিও: {output_video}")
        logger.info(f" ২. মূল ট্রান্সক্রিপ্ট: {raw_txt_path}")
        logger.info(f" ৩. অনুবাদ এনালাইসিস ফাইল: {analysis_txt_path}")
        logger.info("==================================================")

    except Exception as e:
        logger.critical(f"❌ পাইপলাইন এক্সিকিউশনে সমস্যা ঘটেছে: {str(e)}")
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
