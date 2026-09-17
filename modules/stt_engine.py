import os
import re
from faster_whisper import WhisperModel
from config.settings import Config
from utils.gpu import clear_vram
from utils.logger import logger

def transcribe_english(vocal_audio_path: str) -> list[dict]:
    model = WhisperModel(
        Config.WHISPER_MODEL_SIZE, 
        device=Config.DEVICE, 
        compute_type=Config.COMPUTE_TYPE
    )
    
    # vad_filter=True যুক্ত করায় নীরবতা ও ব্যাকগ্রাউন্ড মিউজিক বাদ যাবে
    segments, _ = model.transcribe(
        vocal_audio_path, 
        language="en", 
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    results = []
    for segment in segments:
        text = segment.text.strip()
        # মিউজিক বা অহেতুক হ্যালুসিনেশন ফিল্টার
        if text and not re.match(r'^(is\s*)+$', text, re.IGNORECASE):
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": text
            })
    
    del model
    clear_vram()
    return results

def group_into_full_sentences(segments: list[dict], max_gap: float = 0.8, max_duration: float = 12.0) -> list[dict]:
    if not segments:
        return []

    grouped = []
    curr_text = ""
    curr_start = segments[0]["start"]
    curr_end = segments[0]["end"]

    for seg in segments:
        text = seg["text"].strip()
        start = seg["start"]
        end = seg["end"]

        gap = start - curr_end
        duration = end - curr_start
        is_sentence_end = bool(re.search(r'[.!?]$', curr_text))

        if curr_text and (gap > max_gap or is_sentence_end or duration > max_duration):
            grouped.append({
                "start": curr_start,
                "end": curr_end,
                "text": curr_text.strip()
            })
            curr_text = text
            curr_start = start
            curr_end = end
        else:
            if curr_text:
                curr_text += " " + text
            else:
                curr_text = text
            curr_end = end

    if curr_text:
        grouped.append({
            "start": curr_start,
            "end": curr_end,
            "text": curr_text.strip()
        })

    logger.info(f"স্মার্ট সেগমেন্টেশন: {len(segments)} টি টুকরো → {len(grouped)} টি বাক্যে রূপান্তরিত।")
    return grouped

def save_transcript_to_txt(segments: list[dict], output_path: str):
    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("      ORIGINAL ENGLISH TRANSCRIPTION (WHISPER)    \n")
            f.write("==================================================\n\n")
            
            for idx, seg in enumerate(segments, 1):
                start = seg.get('start', 0.0)
                end = seg.get('end', 0.0)
                duration = end - start
                text = seg.get('text', '')
                
                f.write(f"[{idx:03d}] TIME: {start:.2f}s -> {end:.2f}s (Duration: {duration:.2f}s)\n")
                f.write(f"TEXT : {text}\n")
                f.write("-" * 50 + "\n")
                
        logger.info(f"ইংরেজি ট্রান্সক্রিপ্ট সফলভাবে সেভ হয়েছে: {output_path}")
    except Exception as e:
        logger.error(f"ট্রান্সক্রিপ্ট সেভ করতে সমস্যা হয়েছে: {e}")

def merge_short_segments(segments: list[dict], min_duration: float = 1.5) -> list[dict]:
    return group_into_full_sentences(segments)
