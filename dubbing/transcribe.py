from pathlib import Path
from typing import List, Dict
from faster_whisper import WhisperModel, BatchedInferencePipeline
from dubbing.memory import flush_memory


def transcribe_audio(
    audio_path: Path,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "int8_float16",
    batch_size: int = 16,
) -> List[Dict]:
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    pipeline = BatchedInferencePipeline(model=model)

    segments, _ = pipeline.transcribe(
        str(audio_path),
        language="en",
        condition_on_previous_text=False,
        word_timestamps=False,
        beam_size=1,
        vad_filter=True,
        batch_size=batch_size,
    )

    results = []
    for segment in segments:
        results.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
            "words": [],
        })

    del pipeline
    del model
    flush_memory()
    return results