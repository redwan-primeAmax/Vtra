import subprocess
import json
import logging
from pathlib import Path
from dubbing.errors import InputValidationError, AudioExtractionError
from dubbing.memory import flush_memory

logger = logging.getLogger("dubbing.media")


# ---------------------------------------------------------------
# ONNX Runtime provider ডিটেকশন
# ---------------------------------------------------------------
def check_onnx_gpu() -> bool:
    """ONNX Runtime CUDA provider পাচ্ছে কিনা চেক করে (এবং লগ দেয়)।"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"🔧 ONNX Runtime version: {ort.__version__}")
        logger.info(f"🔧 Available providers: {providers}")

        if "CUDAExecutionProvider" in providers:
            logger.info("✅ MDX-Net GPU (CUDA) তে চলবে")
            return True

        logger.warning(
            "⚠️ CUDAExecutionProvider নেই — MDX-Net CPU-তে চলবে (অনেক ধীর)।\n"
            "   ঠিক করতে:  !pip uninstall -y onnxruntime onnxruntime-gpu\n"
            "              !pip install onnxruntime-gpu"
        )
        return False
    except Exception as e:
        logger.warning(f"⚠️ ONNX Runtime চেক করা যায়নি: {e}")
        return False


# ---------------------------------------------------------------
# Video/audio validation
# ---------------------------------------------------------------
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


# ---------------------------------------------------------------
# Audio + BGM separation
# ---------------------------------------------------------------
def extract_audio_and_bgm(
    video_path: Path,
    work_dir: Path,
    mdx_model: str = "UVR-MDX-NET-Inst_HQ_3.onnx",
    cpu_fallback_model: str = "UVR-MDX-NET-Inst_Main.onnx",
) -> tuple[Path, Path]:
    """16kHz mono WAV এক্সট্র্যাক্ট + MDX-Net দিয়ে vocal/BGM আলাদা।"""

    # ১) raw audio বের করো
    raw_wav = work_dir / "extracted_16k.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(raw_wav),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise AudioExtractionError(f"অডিও এক্সট্র্যাকশন ত্রুটি: {res.stderr}")

    # ২) ONNX GPU check (early — model load করার আগেই)
    has_gpu = check_onnx_gpu()

    # ৩) MDX-Net separation
    mdx_out = work_dir / "mdx_out"
    mdx_out.mkdir(exist_ok=True)
    models_dir = work_dir / "mdx_models"
    models_dir.mkdir(exist_ok=True)

    bgm_wav = raw_wav  # ফলব্যাক

    # GPU না থাকলে হালকা মডেল বেছে নাও
    model_to_use = mdx_model
    if not has_gpu and "HQ" in mdx_model:
        logger.info(f"🐢 CPU mode: '{mdx_model}' থেকে হালকা মডেলে সুইচ → '{cpu_fallback_model}'")
        model_to_use = cpu_fallback_model

    try:
        from audio_separator.separator import Separator

        separator = Separator(
            output_dir=str(mdx_out),
            model_file_dir=str(models_dir),
        )
        logger.info(f"⏳ MDX-Net মডেল লোড: {model_to_use}")
        separator.load_model(model_filename=model_to_use)
        logger.info("⏳ Separation শুরু...")
        output_files = separator.separate(str(raw_wav))

        for f in output_files:
            name = Path(f).name.lower()
            if "instrumental" in name or "no_vocals" in name or "accompaniment" in name:
                candidate = Path(f) if Path(f).is_absolute() else mdx_out / f
                if candidate.exists():
                    bgm_wav = candidate
                    logger.info(f"✅ BGM পাওয়া গেছে: {candidate.name}")
                    break

        # কোনো নাম ম্যাচ না করলে প্রথম আউটপুটকেই BGM ধরো
        if bgm_wav == raw_wav and output_files:
            candidate = Path(output_files[0]) if Path(output_files[0]).is_absolute() else mdx_out / output_files[0]
            if candidate.exists():
                bgm_wav = candidate
                logger.info(f"✅ BGM (fallback): {candidate.name}")

    except Exception as e:
        logger.warning(f"⚠️ MDX-Net সেপারেশন ব্যর্থ, raw WAV-ই BGM হিসেবে ব্যবহার হবে: {e}")

    flush_memory()
    return raw_wav, bgm_wav
