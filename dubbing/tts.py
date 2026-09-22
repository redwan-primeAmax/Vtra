import asyncio
import edge_tts
import soundfile as sf
import numpy as np
import re
from pathlib import Path
from typing import List, Dict
from dubbing.memory import flush_memory

def normalize_bengali_text(text: str) -> str:
    num_map = {'0':'শূন্য', '1':'এক', '2':'দুই', '3':'তিন', '4':'চার', '5':'পাঁচ', '6':'ছয়', '7':'সাত', '8':'আট', '9':'নয়'}
    for k, v in num_map.items():
        text = text.replace(k, v)
    cleaned = re.sub(r'[^\u0980-\u09FF\s।,!?]', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

async def generate_single_tts(text: str, voice: str, output_path: Path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))

def synthesize_speech(items: List[Dict], output_dir: Path, voice: str = "bn-BD-NabanitaNeural") -> List[Dict]:
    """
    Microsoft Edge-TTS ব্যবহার করে সাবলীল ও প্রাকৃতিক বাংলা কণ্ঠস্বর তৈরি করে।
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for idx, item in enumerate(items):
        clean_text = normalize_bengali_text(item["tgt_text"])
        wav_path = output_dir / f"seg_{idx:04d}.wav"
        mp3_path = output_dir / f"seg_{idx:04d}.mp3"
        expected_dur = max(item["end"] - item["start"], 0.5)

        if not clean_text:
            silent_samples = int(24000 * expected_dur)
            output = np.zeros(silent_samples, dtype=np.float32)
            sf.write(str(wav_path), output, samplerate=24000)
        else:
            try:
                asyncio.run(generate_single_tts(clean_text, voice, mp3_path))
                data, sr = sf.read(str(mp3_path))
                sf.write(str(wav_path), data, sr)
                if mp3_path.exists(): mp3_path.unlink()
            except Exception:
                silent_samples = int(24000 * expected_dur)
                output = np.zeros(silent_samples, dtype=np.float32)
                sf.write(str(wav_path), output, samplerate=24000)

        results.append({
            **item,
            "tts_wav": str(wav_path)
        })

    flush_memory()
    return results
