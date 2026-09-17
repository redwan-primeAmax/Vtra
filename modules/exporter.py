import subprocess
import torch
from config.settings import Config
from utils.logger import logger

def merge_audio_tracks(tts_audio_path: str, bgm_path: str, output_audio_path: str):
    """নতুন বাংলা টিটিএস অডিও এবং ব্যাকগ্রাউন্ড মিউজিক (BGM) একসাথে মার্জ করার ফাংশন"""
    logger.info("নতুন বাংলা অডিও এবং ব্যাকগ্রাউন্ড মিউজিক মিক্স করা হচ্ছে...")
    
    cmd = [
        'ffmpeg', '-y',
        '-i', tts_audio_path,
        '-i', bgm_path,
        '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[a]',
        '-map', '[a]',
        '-c:a', 'pcm_s16le',
        output_audio_path
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError("অডিও মার্জ করার প্রক্রিয়ায় এরর দেখা দিয়েছে!")
    logger.info("অডিও মিক্সিং সফলভাবে সম্পন্ন হয়েছে।")

def render_final_video(input_video: str, final_audio: str, output_video: str):
    """GPU NVENC ব্যবহার করে আল্ট্রাফাস্ট ভিডিও এবং ফাইনাল অডিও রেন্ডার করার ফাংশন"""
    use_gpu = torch.cuda.is_available()
    v_codec = 'h264_nvenc' if use_gpu else 'libx264'
    preset_option = 'p1' if use_gpu else 'ultrafast'
    
    logger.info(f"ফাইনাল ভিডিও রেন্ডারিং শুরু হচ্ছে (Encoder: {v_codec}, GPU Mode: {use_gpu})...")
    
    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda' if use_gpu else 'auto',
        '-i', input_video,
        '-i', final_audio,
        '-c:v', v_codec,
        '-preset', preset_option,
        '-pix_fmt', getattr(Config, 'PIXEL_FORMAT', 'yuv420p'),
        '-c:a', getattr(Config, 'AUDIO_CODEC', 'aac'),
        '-b:a', getattr(Config, 'AUDIO_BITRATE', '192k'),
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-movflags', '+faststart',
        output_video
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError("ফাইনাল ভিডিও রেন্ডারিং প্রক্রিয়ায় এরর দেখা দিয়েছে!")
    logger.info("ফাইনাল ভিডিও রেন্ডারিং সফলভাবে সম্পন্ন হয়েছে!")