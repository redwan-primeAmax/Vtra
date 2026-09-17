import os
import glob
import subprocess
import torch
from utils.logger import logger

def scan_and_prepare_input(input_dir="inputs", temp_dir="temp") -> str:
    """যেকোনো ইনপুট ফাইল রিড করবে এবং GPU NVENC ব্যবহার করে সুপারফাস্ট MP4 এ কনভার্ট করবে"""
    logger.info("ইনপুট ডিরেক্টরি স্ক্যান করা হচ্ছে...")
    supported_exts = ['*.webm', '*.mkv', '*.avi', '*.mov', '*.flv', '*.mp4']
    files = []
    
    for ext in supported_exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not files:
        logger.error(f"ধাপ ব্যর্থ: '{input_dir}' ডিরেক্টরিতে কোনো ভিডিও ফাইল পাওয়া যায়নি!")
        raise FileNotFoundError("কোনো ইনপুট ভিডিও পাওয়া যায়নি।")
        
    raw_file = files[0]
    file_ext = os.path.splitext(raw_file)[1].lower()
    logger.info(f"ইনপুট ফাইল সনাক্ত করা হয়েছে: '{raw_file}' (ফরম্যাট: {file_ext})")
    
    converted_path = os.path.join(temp_dir, "normalized_input.mp4")
    
    # GPU সাপোর্ট চেক
    use_gpu = torch.cuda.is_available()
    v_codec = 'h264_nvenc' if use_gpu else 'libx264'
    preset_option = 'p1' if use_gpu else 'ultrafast'
    
    logger.info(f"ভিডিও কনভার্সন চালিত হচ্ছে (Encoder: {v_codec}, GPU Mode: {use_gpu})...")
    
    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda' if use_gpu else 'auto',
        '-i', raw_file,
        '-c:v', v_codec,
        '-preset', preset_option,
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        converted_path
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    process.wait()
    
    if process.returncode == 0:
        logger.info("GPU হাই-স্পিড ভিডিও কনভার্সন সম্পন্ন হয়েছে!")
        return converted_path
    else:
        logger.error("NVENC ব্যর্থ হয়েছে, CPU এনকোডিং ব্যবহার করা হচ্ছে...")
        cmd_cpu = ['ffmpeg', '-y', '-i', raw_file, '-c:v', 'libx264', '-preset', 'ultrafast', converted_path]
        subprocess.run(cmd_cpu, check=True)
        return converted_path

def extract_audio(video_path: str, output_audio: str):
    """ভিডিও থেকে হাই-স্পিড অডিও এক্সট্র্যাক্ট করার ফাংশন (যেটি মিসিং ছিল)"""
    logger.info("ভিডিও থেকে অরিজিনাল অডিও এক্সট্র্যাক্ট করা হচ্ছে...")
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        output_audio
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("অডিও এক্সট্র্যাক্ট সফল হয়েছে।")
