from pathlib import Path
from typing import List, Dict
from faster_whisper import WhisperModel
from dubbing.memory import flush_memory

def transcribe_audio(
    audio_path: Path, 
    model_size: str = "large-v3", 
    device: str = "cuda", 
    compute_type: str = "float16"
) -> List[Dict]:
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, _ = model.transcribe(
        str(audio_path),
        language="en",
        condition_on_previous_text=False,
        word_timestamps=True,
        beam_size=5
    )

    results = []
    for segment in segments:
        words = []
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3)
                })
        results.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
            "words": words
        })

    del model
    flush_memory()
    return results
