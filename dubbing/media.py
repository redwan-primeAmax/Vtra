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
        "-show_format", "-show_streams", str(video_path),
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


def extract_audio_only(video_path: Path, work_dir: Path) -> Path:
    """শুধু 16kHz mono WAV এক্সট্র্যাক্ট — transcribe-only মোডের জন্য।"""
    raw_wav = work_dir / "extracted_16k.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(raw_wav),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise AudioExtractionError(f"অডিও এক্সট্র্যাকশন ত্রুটি: {res.stderr}")
    return raw_wav


def extract_audio_and_bgm(video_path: Path, work_dir: Path) -> tuple[Path, Path]:
    raw_wav = work_dir / "extracted_16k.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(raw_wav),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise AudioExtractionError(f"অডিও এক্সট্র্যাকশন ত্রুটি: {res.stderr}")

    # ---------- Spleeter (2-stem: vocals + accompaniment) ----------
    spleeter_out = work_dir / "spleeter_out"
    spleeter_out.mkdir(exist_ok=True)

    spleeter_cmd = [
        "spleeter", "separate",
        "-p", "spleeter:2stems",
        "-o", str(spleeter_out),
        str(raw_wav),
    ]
    subprocess.run(spleeter_cmd, capture_output=True, text=True)

    # Spleeter আউটপুট: <spleeter_out>/<input_stem>/accompaniment.wav
    bgm_wav = spleeter_out / raw_wav.stem / "accompaniment.wav"
    if not bgm_wav.exists():
        # কোনো কারণে না পেলে raw_wav-ই BGM হিসেবে ব্যবহার
        bgm_wav = raw_wav

    flush_memory()
    return raw_wav, bgm_wav