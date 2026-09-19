import logging
import shutil
from pathlib import Path
from dubbing.config import DubbingConfig
from dubbing.pipeline import DubbingPipeline
from dubbing.errors import DubbingError

# লগিং কনফিগারেশন
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")

def process_videos():
    # Google Drive পাথ
    drive_input_dir = Path("/content/drive/MyDrive/Video/input")
    drive_output_dir = Path("/content/drive/MyDrive/Video/output")
    
    if not drive_input_dir.exists():
        drive_input_dir = Path("/content/drive/Video/input")
        drive_output_dir = Path("/content/drive/Video/output")

    drive_input_dir.mkdir(parents=True, exist_ok=True)
    drive_output_dir.mkdir(parents=True, exist_ok=True)

    # স্থানীয় Colab ডিরেক্টরি (সর্বোচ্চ স্পিডের জন্য)
    local_base_dir = Path("/content/local_workspace")
    local_input_dir = local_base_dir / "input"
    local_output_dir = local_base_dir / "output"
    
    local_input_dir.mkdir(parents=True, exist_ok=True)
    local_output_dir.mkdir(parents=True, exist_ok=True)

    # ড্রাইভে ভিডিও ফাইল খোঁজা
    video_files = list(drive_input_dir.glob("*.mp4")) + list(drive_input_dir.glob("*.mkv"))

    if not video_files:
        logger.warning(f"কোনো ভিডিও ফাইল পাওয়া যায়নি! ফাইল রাখুন এখানে: {drive_input_dir}")
        return

    logger.info(f"মোট {len(video_files)} টি ভিডিও প্রসেস করা হবে।")

    for index, drive_video_path in enumerate(video_files, start=1):
        filename = drive_video_path.name
        stem = drive_video_path.stem
        
        logger.info(f"\n==========================================")
        logger.info(f"[{index}/{len(video_files)}] ফাইল শনাক্ত হয়েছে: {filename}")
        logger.info(f"==========================================")

        # ১. ফাইল Colab-এর লোকাল এনভায়রনমেন্টে কপি করা (Fast I/O)
        local_video_path = local_input_dir / filename
        logger.info(f"Colab লোকাল স্টোরেজে ফাইল কপি করা হচ্ছে: {filename}...")
        shutil.copy2(drive_video_path, local_video_path)

        local_output_video = local_output_dir / f"dubbed_{stem}.mp4"
        local_work_dir = local_base_dir / f"work_{stem}"

        # ২. কনফিগারেশন সেটআপ (লোকাল পাথে)
        config = DubbingConfig(
            input_video=local_video_path,
            output_video=local_output_video,
            work_dir=local_work_dir,
            compute_type="float16"
        )

        try:
            # ৩. ডাবিং পাইপলাইন চালানো
            pipeline = DubbingPipeline(config)
            pipeline.run()

            # ৪. প্রসেস করা শেষ হলে আউটপুট ফাইল ড্রাইভ-এ ব্যাকআপ নেওয়া
            drive_final_output = drive_output_dir / f"dubbed_{stem}.mp4"
            logger.info(f"আউটপুট ফাইল Google Drive-এ ট্রান্সফার করা হচ্ছে: {drive_final_output.name}")
            shutil.copy2(local_output_video, drive_final_output)

            logger.info(f"সফলভাবে ড্রাইভে সেভ হয়েছে: {drive_final_output}")

        except DubbingError as e:
            logger.error(f"ডাবিং পাইপলাইনে ত্রুটি [{filename}]: {str(e)}")
        except Exception as e:
            logger.error(f"অপ্রত্যাশিত ত্রুটি [{filename}]: {str(e)}", exc_info=True)
        
        finally:
            # ৫. ডিস্ক স্পেস খালি করতে লোকাল অস্থায়ী ফাইল মুছে ফেলা
            if local_video_path.exists():
                local_video_path.unlink()
            if local_output_video.exists():
                local_output_video.unlink()
            if local_work_dir.exists():
                shutil.rmtree(local_work_dir, ignore_errors=True)
            logger.info("লোকাল টেম্পোরারি ফাইল পরিষ্কার করা হয়েছে।")

if __name__ == "__main__":
    process_videos()
