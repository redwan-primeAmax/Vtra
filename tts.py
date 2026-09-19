import torch
import soundfile as sf
import re
from pathlib import Path
from typing import List, Dict
from transformers import VitsModel, AutoTokenizer
from dubbing.memory import flush_memory

def normalize_bengali_text(text: str) -> str:
    num_map = {'0':'শূন্য', '1':'এক', '2':'দুই', '3':'তিন', '4':'চার', '5':'পাঁচ', '6':'ছয়', '7':'সাত', '8':'আট', '9':'নয়'}
    for k, v in num_map.items():
        text = text.replace(k, v)
    return re.sub(r'[^\u0980-\u09FF\s।,!?]', '', text)

def synthesize_speech(items: List[Dict], output_dir: Path, model_name: str = "facebook/mms-tts-ben") -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VitsModel.from_pretrained(model_name).to(device)

    results = []
    for idx, item in enumerate(items):
        clean_text = normalize_bengali_text(item["tgt_text"])
        if not clean_text.strip():
            clean_text = "নীরবতা"

        inputs = tokenizer(clean_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs).waveform.cpu().squeeze().numpy()

        wav_path = output_dir / f"seg_{idx:04d}.wav"
        sf.write(str(wav_path), output, samplerate=model.config.sampling_rate)

        results.append({
            **item,
            "tts_wav": str(wav_path),
            "sample_rate": model.config.sampling_rate
        })

    del model
    del tokenizer
    flush_memory()
    return results
