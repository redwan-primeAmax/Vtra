import subprocess
import json
from pathlib import Path
from dubbing.errors import InputValidationError, AudioExtractionError
from dubbing.memory import flush_memory

def validate_input_file(video_path: Path) -> dict:
    if not video_path.exists():
        raise InputValidationError(f"ইনপুট ফাইল পাওয়া যায়নি: {video_path}")

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise InputValidationError(f"ffprobe ব্যর্থ হয়েছে: {res.stderr}")

    info = json.loads(res.stdout)
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    has_video = any(s.get("codec_type") == "video" for s in info.get("streams", []))

    if not has_video or not has_audio:
        raise InputValidationError("ভিডিও বা অডিও স্ট্রিম পাওয়া যায়নি।")

    return info

def extract_audio_and_bgm(video_path: Path, work_dir: Path) -> tuple[Path, Path]:
    raw_wav = work_dir / "extracted_16k.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(raw_wav)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise AudioExtractionError(f"অডিও এক্সট্র্যাকশন ত্রুটি: {res.stderr}")

    demucs_out = work_dir / "demucs_out"
    demucs_cmd = [
        "demucs", "--two-stems=vocals", "-n", "htdemucs",
        "-o", str(demucs_out), str(raw_wav)
    ]
    subprocess.run(demucs_cmd, capture_output=True, text=True)

    bgm_wav = demucs_out / "htdemucs" / "extracted_16k" / "no_vocals.wav"
    if not bgm_wav.exists():
        bgm_wav = raw_wav

    flush_memory()
    return raw_wav, bgm_wav
