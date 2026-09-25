# ⚠️ সবার আগে runner import — env + logging setup হবে
from dubbing import runner  # noqa: F401

import logging
import shutil
from pathlib import Path

from dubbing.config import DubbingConfig
from dubbing.pipeline import DubbingPipeline
from dubbing.errors import DubbingError

logger = logging.getLogger("main")


def process_videos():
    # ১. পাথ কনফিগারেশন
    drive_input_dir = Path("/content/drive/MyDrive/Video/input")
    drive_output_dir = Path("/content/drive/MyDrive/Video/output")

    if not drive_input_dir.exists():
        drive_input_dir = Path("/content/drive/Video/input")
        drive_output_dir = Path("/content/drive/Video/output")

    drive_input_dir.mkdir(parents=True, exist_ok=True)
    drive_output_dir.mkdir(parents=True, exist_ok=True)

    local_base_dir = Path("/content/local_workspace")
    local_input_dir = local_base_dir / "input"
    local_output_dir = local_base_dir / "output"

    local_input_dir.mkdir(parents=True, exist_ok=True)
    local_output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [f for f in drive_input_dir.iterdir() if f.is_file()]

    custom_glossary = [
        "Ballon d'Or", "Champions League", "Premier League",
        "Real Madrid", "Barcelona", "Messi", "Ronaldo"
    ]

    for index, drive_video_path in enumerate(video_files, start=1):
        filename = drive_video_path.name
        stem = drive_video_path.stem

        print(f"\n==========================================")
        print(f"🎬 প্রসেসিং শুরু হচ্ছে ({index}/{len(video_files)}): {filename}")
        print(f"==========================================")

        local_video_path = local_input_dir / filename
        shutil.copy2(drive_video_path, local_video_path)

        local_output_video = local_output_dir / f"dubbed_{stem}.mp4"
        local_work_dir = local_base_dir / f"work_{stem}"

        config = DubbingConfig(
            input_video=local_video_path,
            output_video=local_output_video,
            work_dir=local_work_dir,
            tts_voice="bn-BD-NabanitaNeural",
            glossary=custom_glossary
        )

        try:
            pipeline = DubbingPipeline(config)
            pipeline.run()

            drive_final_output = drive_output_dir / f"dubbed_{stem}.mp4"
            shutil.copy2(local_output_video, drive_final_output)
            logger.info(f"✅ ড্রাইভে সেভ হয়েছে: {drive_final_output.name}")

        except DubbingError as e:
            logger.error(f"❌ পাইপলাইন ত্রুটি [{filename}]: {str(e)}")
        except Exception as e:
            logger.error(f"❌ অপ্রত্যাশিত ত্রুটি [{filename}]: {str(e)}")
        finally:
            if local_video_path.exists(): local_video_path.unlink()
            if local_output_video.exists(): local_output_video.unlink()
            if local_work_dir.exists(): shutil.rmtree(local_work_dir, ignore_errors=True)


if __name__ == "__main__":
    process_videos()