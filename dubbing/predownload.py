"""
ব্যাকগ্রাউন্ডে Demucs ও Whisper মডেল ডাউনলোড করার ম্যানেজার।
main.py শুরুতেই থ্রেড চালু হয়, তারপর পাইপলাইন নিজের কাজ করতে থাকে।
"""
import threading
import logging

logger = logging.getLogger("dubbing.predownload")

_started = set()
_lock = threading.Lock()


def _download_demucs():
    try:
        from demucs.pretrained import get_model
        model = get_model("htdemucs")
        del model
        logger.info("✅ [BG] Demucs htdemucs প্রস্তুত")
    except Exception as e:
        logger.error(f"❌ [BG] Demucs ডাউনলোড ব্যর্থ: {e}")


def _download_whisper(model_size: str = "large-v3"):
    try:
        from huggingface_hub import snapshot_download
        repo = f"Systran/faster-whisper-{model_size}"
        snapshot_download(repo_id=repo)
        logger.info(f"✅ [BG] Whisper {model_size} প্রস্তুত")
    except Exception as e:
        logger.error(f"❌ [BG] Whisper ডাউনলোড ব্যর্থ: {e}")


def _spawn(name: str, target, *args):
    with _lock:
        if name in _started:
            return
        _started.add(name)
    t = threading.Thread(target=target, args=args, daemon=True, name=f"dl-{name}")
    t.start()
    logger.info(f"🚀 [BG] {name} ডাউনলোড শুরু হয়েছে")


def start_all_downloads(whisper_model: str = "large-v3"):
    """পাইপলাইনের একদম শুরুতে কল করুন — ২টি থ্রেড চালু হবে।"""
    _spawn("demucs", _download_demucs)
    _spawn("whisper", _download_whisper, whisper_model)