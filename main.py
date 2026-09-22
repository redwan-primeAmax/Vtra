import logging
import shutil
from pathlib import Path
from dubbing.config import DubbingConfig
from dubbing.pipeline import DubbingPipeline
from dubbing.errors import DubbingError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("main")

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".wmv", ".m4v", ".ts"}

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

    video_files = [
        f for f in drive_input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not video_files:
        logger.warning(f"কোনো সামঞ্জস্যপূর্ণ ভিডিও ফাইল পাওয়া যায়নি! সাপোর্টেড ফরম্যাট: {SUPPORTED_EXTENSIONS}")
        return

    logger.info(f"মোট {len(video_files)} টি ভিডিও প্রসেস করা হবে।")

    for index, drive_video_path in enumerate(video_files, start=1):
        filename = drive_video_path.name
        stem = drive_video_path.stem
        
        logger.info(f"\n==========================================")
        logger.info(f"[{index}/{len(video_files)}] প্রসেসিং শুরু: {filename}")
        logger.info(f"==========================================")

        local_video_path = local_input_dir / filename
        shutil.copy2(drive_video_path, local_video_path)

        local_output_video = local_output_dir / f"dubbed_{stem}.mp4"
        local_work_dir = local_base_dir / f"work_{stem}"

        config = DubbingConfig(
            input_video=local_video_path,
            output_video=local_output_video,
            work_dir=local_work_dir,
            compute_type="float16"
        )

        try:
            pipeline = DubbingPipeline(config)
            pipeline.run()

            drive_final_output = drive_output_dir / f"dubbed_{stem}.mp4"
            shutil.copy2(local_output_video, drive_final_output)
            logger.info(f"সফলভাবে ড্রাইভে সেভ হয়েছে: {drive_final_output}")

        except DubbingError as e:
            logger.error(f"ডাবিং পাইপলাইনে ত্রুটি [{filename}]: {str(e)}")
        except Exception as e:
            logger.error(f"অপ্রত্যাশিত ত্রুটি [{filename}]: {str(e)}", exc_info=True)
        
        finally:
            if local_video_path.exists(): local_video_path.unlink()
            if local_output_video.exists(): local_output_video.unlink()
            if local_work_dir.exists(): shutil.rmtree(local_work_dir, ignore_errors=True)

if __name__ == "__main__":
    process_videos(
