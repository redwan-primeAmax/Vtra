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
    elif isinstance(seg, dict):
        text = seg.get('translated_text', seg.get('text', ''))
    else:
        text = str(seg)

    if not text or not str(text).strip():
        logger.warning(f"সেগমেন্ট {idx}: খালি টেক্সট থাকায় স্কিপ করা হলো।")
        return None

    # অডিওর স্পিড স্বয়ংক্রিয়ভাবে অস্বাভাবিক না বাড়ার জন্য ফিক্সড স্বাভাবিক কন্ঠস্বর নির্ধারণ[span_1](start_span)[span_1](end_span)
    rate_str = "+0%[span_2](start_span)"[span_2](end_span)

    try:
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        asyncio.run(_generate_audio_single(str(text), voice, target_file, rate_str=rate_str))[span_3](start_span)[span_3](end_span)
        logger.info(f"  [TTS সেগমেন্ট {idx:02d}] টেক্সট: '{text[:25]}...' | প্রয়োগকৃত স্পিড: {rate_str}")[span_4](start_span)[span_4](end_span)
        return target_file[span_5](start_span)[span_5](end_span)
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
