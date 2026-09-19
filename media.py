import subprocess
import json
from pathlib import Path
from dubbing.errors import InputValidationError, AudioExtractionError

def validate_input_file(video_path: Path) -> dict:
    if not video_path.exists():
        raise InputValidationError(f"ইনপুট ফাইল পাওয়া যায়নি: {video_path}")

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise InputValidationError(f"ffprobe ব্যর্থ হয়েছে: {res.stderr}")

    info = json.loads(res.stdout)
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    has_video = any(s.get("codec_type") == "video" for s in info.get("streams", []))

    if not has_video or not has_audio:
        raise InputValidationError("ইনপুট ফাইলে ভ্যালিড ভিডিও অথবা অডিও স্ট্রিম পাওয়া যায়নি।")

    return info

def extract_audio(video_path: Path, output_wav: Path) -> Path:
    """.webm ফাইলসহ সমস্ত কোডেড ভিডিও থেকে নিরাপদে ১৬kHz মনো অডিও এক্সট্র্যাক্ট করে।"""
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",              # টাইমিং ফ্রেম ফিক্স করার জন্য
        "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-avoid_negative_ts", "make_zero", # নেগেটিভ টাইমস্ট্যাম্প ফিক্স
        str(output_wav)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise AudioExtractionError(f"অডিও এক্সট্র্যাকশন ত্রুটি: {res.stderr}")
    return output_wav
