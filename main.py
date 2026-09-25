# ⚠️ সবার আগে runner import — env + logging setup
from dubbing import runner  # noqa: F401

import logging
import shutil
from pathlib import Path

from dubbing.config import DubbingConfig
from dubbing.pipeline import DubbingPipeline
from dubbing.errors import DubbingError
from dubbing.media import extract_audio_only
from dubbing.transcribe import transcribe_audio
from dubbing.script_io import (
    find_video_file,
    find_translated_file,
    load_translated_segments,
    save_transcribed_json,
    save_ai_markdown,
)
from dubbing.memory import flush_memory

logger = logging.getLogger("main")

DRIVE_INPUT_PRIMARY = Path("/content/drive/MyDrive/Video/input")
DRIVE_INPUT_FALLBACK = Path("/content/drive/Video/input")
DRIVE_OUTPUT_PRIMARY = Path("/content/drive/MyDrive/Video/output")
DRIVE_OUTPUT_FALLBACK = Path("/content/drive/Video/output")

LOCAL_BASE = Path("/content/local_workspace")

CUSTOM_GLOSSARY = [
    "Ballon d'Or", "Champions League", "Premier League",
    "Real Madrid", "Barcelona", "Messi", "Ronaldo",
]


def _resolve_drive_dirs():
    if DRIVE_INPUT_PRIMARY.exists():
        return DRIVE_INPUT_PRIMARY, DRIVE_OUTPUT_PRIMARY
    return DRIVE_INPUT_FALLBACK, DRIVE_OUTPUT_FALLBACK


def _transcribe_only(folder: Path, video_path: Path):
    """translated.json নেই → শুধু transcribe, JSON + AI markdown সেভ।"""
    print("📝 translated.json নেই — শুধু transcribe হবে")
    work_dir = LOCAL_BASE / f"transcribe_{video_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    local_video = work_dir / video_path.name
    shutil.copy2(video_path, local_video)

    try:
        raw_wav = extract_audio_only(local_video, work_dir)
        segments = transcribe_audio(
            raw_wav,
            model_size="large-v3",
            device="cuda",
            compute_type="float16",
        )
        flush_memory()

        if not segments:
            print("❌ কোনো ট্রান্সক্রিপ্ট পাওয়া যায়নি")
            return

        save_transcribed_json(folder, video_path.stem, segments)
        save_ai_markdown(folder, video_path.stem, segments, CUSTOM_GLOSSARY)

        print(f"\n✅ সম্পন্ন — এখন অনুবাদ করুন:")
        print(f"   📄 {video_path.stem}.for_ai.md → ChatGPT/Gemini-এ পেস্ট করুন")
        print(f"   📄 উত্তর সংরক্ষণ করুন: {video_path.stem}.translated.json")
        print(f"   📌 তারপর আবার এই নোটবুক চালান")
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def _full_pipeline(folder: Path, video_path: Path, translated_path: Path, drive_output: Path):
    """translated.json আছে → TTS + Mixer চালাও।"""
    print(f"✅ translated.json পাওয়া গেছে: {translated_path.name}")

    pre_translated = load_translated_segments(translated_path)
    if not pre_translated:
        print("❌ translated.json খালি — বাদ")
        return

    print(f"   মোট {len(pre_translated)} সেগমেন্ট")

    work_dir = LOCAL_BASE / f"work_{video_path.stem}"
    local_video = work_dir / video_path.name
    local_output = LOCAL_BASE / "output" / f"dubbed_{video_path.stem}.mp4"
    work_dir.mkdir(parents=True, exist_ok=True)
    local_output.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(video_path, local_video)

    config = DubbingConfig(
        input_video=local_video,
        output_video=local_output,
        work_dir=work_dir,
        tts_voice="bn-BD-NabanitaNeural",
        glossary=CUSTOM_GLOSSARY,
    )

    try:
        pipeline = DubbingPipeline(config, pre_translated_segments=pre_translated)
        pipeline.run()

        final_output = drive_output / folder.name / f"dubbed_{video_path.stem}.mp4"
        final_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_output, final_output)
        print(f"✅ Drive-এ সংরক্ষিত: {final_output}")
    except DubbingError as e:
        print(f"❌ পাইপলাইন ত্রুটি: {e}")
    except Exception as e:
        print(f"❌ অপ্রত্যাশিত ত্রুটি: {e}")
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        if local_output.exists():
            local_output.unlink()


def process_videos():
    drive_input, drive_output = _resolve_drive_dirs()
    drive_input.mkdir(parents=True, exist_ok=True)
    drive_output.mkdir(parents=True, exist_ok=True)
    LOCAL_BASE.mkdir(parents=True, exist_ok=True)

    subfolders = sorted([d for d in drive_input.iterdir() if d.is_dir()])
    if not subfolders:
        print(f"⚠️ কোনো সাবফোল্ডার পাওয়া যায়নি: {drive_input}")
        return

    print(f"📂 মোট {len(subfolders)}টি সাবফোল্ডার পাওয়া গেছে")

    for idx, folder in enumerate(subfolders, 1):
        video_path = find_video_file(folder)
        if not video_path:
            print(f"\n⚠️ [{idx}] '{folder.name}' এ কোনো ভিডিও নেই — বাদ")
            continue

        print(f"\n{'=' * 50}")
        print(f"🎬 [{idx}/{len(subfolders)}] {folder.name} → {video_path.name}")
        print(f"{'=' * 50}")

        translated_path = find_translated_file(folder)

        if translated_path is None:
            _transcribe_only(folder, video_path)
        else:
            _full_pipeline(folder, video_path, translated_path, drive_output)


if __name__ == "__main__":
    process_videos()