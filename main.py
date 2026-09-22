import os
import logging
import shutil
from pathlib import Path
from dubbing.config import DubbingConfig
from dubbing.pipeline import DubbingPipeline
from dubbing.errors import DubbingError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("main")

API_KEY = "YOUR_GEMINI_API_KEY"  # আপনার Gemini API Key এখানে বসান

def process_videos():
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

    for index, drive_video_path in enumerate(video_files, start=1):
        filename = drive_video_path.name
        stem = drive_video_path.stem

        local_video_path = local_input_dir / filename
        shutil.copy2(drive_video_path, local_video_path)

        local_output_video = local_output_dir / f"dubbed_{stem}.mp4"
        local_work_dir = local_base_dir / f"work_{stem}"

        config = DubbingConfig(
            input_video=local_video_path,
            output_video=local_output_video,
            work_dir=local_work_dir,
            api_key=API_KEY,
            tts_voice="bn-BD-NabanitaNeural"
        )

        try:
            pipeline = DubbingPipeline(config)
            pipeline.run()

            drive_final_output = drive_output_dir / f"dubbed_{stem}.mp4"
            shutil.copy2(local_output_video, drive_final_output)
            logger.info(f"ড্রাইভে সেভ হয়েছে: {drive_final_output}")

        except DubbingError as e:
            logger.error(f"ডাবিং পাইপলাইনে ত্রুটি [{filename}]: {str(e)}")
        except Exception as e:
            logger.error(f"অপ্রত্যাশিত ত্রুটি [{filename}]: {str(e)}", exc_info=True)
        finally:
            if local_video_path.exists(): local_video_path.unlink()
            if local_output_video.exists(): local_output_video.unlink()
            if local_work_dir.exists(): shutil.rmtree(local_work_dir, ignore_errors=True)

if __name__ == "__main__":
    process_videos()
