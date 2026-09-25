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

    # ---------- MDX-Net (audio-separator) ----------
    mdx_out = work_dir / "mdx_out"
    mdx_out.mkdir(exist_ok=True)
    models_dir = work_dir / "mdx_models"
    models_dir.mkdir(exist_ok=True)

    bgm_wav = raw_wav  # ফলব্যাক
    try:
        from audio_separator.separator import Separator

        separator = Separator(
            output_dir=str(mdx_out),
            model_file_dir=str(models_dir),
        )
        separator.load_model(model_filename="UVR-MDX-NET-Inst_HQ_3.onnx")
        output_files = separator.separate(str(raw_wav))

        for f in output_files:
            name = Path(f).name.lower()
            if "instrumental" in name or "no_vocals" in name or "accompaniment" in name:
                candidate = Path(f) if Path(f).is_absolute() else mdx_out / f
                if candidate.exists():
                    bgm_wav = candidate
                    break

        # কোনো নাম ম্যাচ না করলে প্রথম আউটপুটকেই BGM ধরুন
        if bgm_wav == raw_wav and output_files:
            candidate = Path(output_files[0]) if Path(output_files[0]).is_absolute() else mdx_out / output_files[0]
            if candidate.exists():
                bgm_wav = candidate

    except Exception as e:
        print(f"⚠️ MDX-Net সেপারেশন ব্যর্থ, raw WAV-ই BGM: {e}")

    flush_memory()
    return raw_wav, bgm_wav