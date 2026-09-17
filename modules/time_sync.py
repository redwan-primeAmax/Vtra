import os
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
from config.settings import Config
from utils.logger import logger

def _get_val(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    elif hasattr(obj, key):
        return getattr(obj, key, default)
    return default

def get_audio_duration(file_path):
    try:
        if not file_path or not os.path.exists(str(file_path)):
            return 0.0
        y, sr = sf.read(str(file_path))
        return len(y) / sr
    except Exception as e:
        logger.error(f"অডিও ডিউরেশন পরিমাপে সমস্যা: {e}")
        return 0.0

def _parse_sync_inputs(*args, **kwargs):
    tts_paths = kwargs.get('tts_audio_paths') or kwargs.get('tts_paths') or kwargs.get('audio_paths')
    segments = kwargs.get('whisper_segments') or kwargs.get('segments')
    output_dir = kwargs.get('output_dir', 'temp/synced')
    output_file = kwargs.get('output_file', 'temp/final_vocals.wav')
    return tts_paths or [], segments or [], output_dir, output_file

class DubbingAudioResult(list):
    def __init__(self, synced_files, merged_file_path):
        super().__init__(synced_files)
        self.synced_files = synced_files
        self.merged_file_path = str(merged_file_path)

    def __str__(self):
        return self.merged_file_path

    def __repr__(self):
        return self.merged_file_path

    def __fspath__(self):
        return self.merged_file_path

def sync_audio_duration(*args, **kwargs):
    tts_audio_paths, whisper_segments, output_dir, _ = _parse_sync_inputs(*args, **kwargs)
    os.makedirs(output_dir, exist_ok=True)
    synced_files = []

    for idx, (audio, seg) in enumerate(zip(tts_audio_paths, whisper_segments)):
        audio_path = audio if isinstance(audio, str) else audio.get('audio_path', '')
        if not audio_path or not os.path.exists(str(audio_path)):
            continue

        start = float(_get_val(seg, 'start', 0.0) or 0.0)
        end = float(_get_val(seg, 'end', 0.0) or 0.0)
        target_duration = end - start
        out_path = os.path.join(output_dir, f"synced_{idx}.wav")

        try:
            y, sr = sf.read(str(audio_path))
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            
            current_duration = len(y) / sr

            if current_duration > 0 and target_duration > 0.1:
                raw_rate = current_duration / target_duration
                clamped_rate = max(0.80, min(raw_rate, 1.25))
                
                y_stretched = pyrb.time_stretch(y, sr, clamped_rate)
                sf.write(out_path, y_stretched, sr)
            else:
                sf.write(out_path, y, sr)

            synced_files.append(out_path)
        except Exception as e:
            logger.error(f"টাইম সিঙ্ক ত্রুটি (সেগমেন্ট {idx}): {e}")
            synced_files.append(str(audio_path))

    return synced_files

def sync_and_merge_segments(*args, **kwargs):
    tts_audio_paths, whisper_segments, output_dir, output_file = _parse_sync_inputs(*args, **kwargs)
    
    synced_files = sync_audio_duration(
        tts_audio_paths=tts_audio_paths,
        whisper_segments=whisper_segments,
        output_dir=output_dir
    )

    target_sr = 44100
    max_end = max([float(_get_val(seg, 'end', 0.0) or 0.0) for seg in whisper_segments], default=300.0)
    total_samples = int((max_end + 10.0) * target_sr)
    combined = np.zeros(total_samples, dtype=np.float32)

    sorted_pairs = sorted(
        zip(synced_files, whisper_segments),
        key=lambda p: float(_get_val(p[1], 'start', 0.0) or 0.0)
    )

    current_timeline_position = 0.0

    for i, (audio_path, seg) in enumerate(sorted_pairs):
        if not os.path.exists(str(audio_path)):
            continue
        try:
            y, sr = sf.read(str(audio_path))
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)

            start = float(_get_val(seg, 'start', 0.0) or 0.0)
            actual_start = max(start, current_timeline_position)
            
            start_sample = int(actual_start * target_sr)
            end_sample = start_sample + len(y)

            if end_sample > len(combined):
                combined = np.pad(combined, (0, end_sample - len(combined)))

            combined[start_sample:end_sample] += y
            current_timeline_position = actual_start + (len(y) / sr) + 0.15

        except Exception as e:
            logger.error(f"অডিও মার্জ ত্রুটি ({audio_path}): {e}")

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    sf.write(output_file, combined, target_sr)
    logger.info(f"চূড়ান্ত মার্জড অডিও প্রস্তুত: {output_file}")
    
    return DubbingAudioResult(synced_files, output_file)
