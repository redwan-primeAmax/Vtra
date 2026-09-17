import os
import asyncio
import numpy as np
import soundfile as sf
import edge_tts
from utils.logger import logger

async def _generate_audio_single(text, voice, output_path, rate_str="+0%"):
    communicate = edge_tts.Communicate(text, voice, rate=rate_str)
    await communicate.save(output_path)

def _process_one_segment(seg, target_file, voice, idx):
    if isinstance(seg, str):
        text = seg
        target_duration = 0.0
    elif isinstance(seg, dict):
        text = seg.get('translated_text', seg.get('text', ''))
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        target_duration = end - start
    else:
        text = str(seg)
        target_duration = 0.0

    if not text or not str(text).strip():
        logger.warning(f"সেগমেন্ট {idx}: খালি টেক্সট থাকায় স্কিপ করা হলো।")
        return None

    words_count = len(str(text).split())
    estimated_tts_time = words_count * 0.35

    rate_str = "+0%"
    if target_duration > 1.0 and estimated_tts_time > 0:
        ratio = estimated_tts_time / target_duration
        if ratio > 1.25:
            rate_pct = min(int((ratio - 1.0) * 40), 20)
            rate_str = f"+{rate_pct}%"
        elif ratio < 0.75:
            rate_pct = min(int((1.0 - ratio) * 40), 20)
            rate_str = f"-{rate_pct}%"

    try:
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        asyncio.run(_generate_audio_single(str(text), voice, target_file, rate_str=rate_str))
        logger.info(f"  [TTS সেগমেন্ট {idx:02d}] টেক্সট: '{text[:25]}...' | প্রয়োগকৃত স্পিড: {rate_str}")
        return target_file
    except Exception as e:
        logger.error(f"TTS জেনারেট করতে সমস্যা (সেগমেন্ট {idx}): {e}")
        return None

def _create_silence_placeholder(duration_sec, output_path):
    sr = 24000
    duration_sec = max(duration_sec, 0.3)
    silence = np.zeros(int(duration_sec * sr), dtype=np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, silence, sr)
    return output_path

def generate_bn_tts(segment_data, output_path="temp", voice="bn-BD-NabanitaNeural"):
    if isinstance(segment_data, list):
        os.makedirs(output_path, exist_ok=True)
        audio_paths = []
        for idx, seg in enumerate(segment_data):
            file_name = f"tts_segment_{idx}.mp3"
            full_path = os.path.join(output_path, file_name)
            res = _process_one_segment(seg, full_path, voice, idx)
            
            if res:
                audio_paths.append(full_path)
            else:
                start = seg.get('start', 0.0) if isinstance(seg, dict) else 0.0
                end = seg.get('end', 0.0) if isinstance(seg, dict) else 0.0
                duration = max(end - start, 0.3)
                silence_path = full_path.replace('.mp3', '_silence.wav')
                _create_silence_placeholder(duration, silence_path)
                audio_paths.append(silence_path)
        
        return audio_paths
    else:
        full_path = os.path.join(output_path, "tts_segment_0.mp3") if os.path.isdir(output_path) else output_path
        res = _process_one_segment(segment_data, full_path, voice, 0)
        return full_path if res else ""

generate_tts_segment = generate_bn_tts
