import subprocess
from config.settings import Config
from utils.logger import logger

def separate_audio(input_audio_path: str, temp_dir: str) -> tuple[str, str]:
    logger.info(f"Demucs GPU স্প্লিটিং শুরু হচ্ছে ({Config.DEVICE} মোডে)...")
    
    cmd = [
        'demucs', '--two-stems=vocals',
        '-n', 'htdemucs',
        '-d', Config.DEVICE,  # Force CUDA
        '-o', temp_dir,
        input_audio_path
    ]
    
    # লাইভ টার্মিনাল আউটপুট প্রিন্ট
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError("Demucs প্রসেসিংয়ে ত্রুটি ঘটেছে!")
        
    import os
    filename = os.path.splitext(os.path.basename(input_audio_path))[0]
    separated_dir = os.path.join(temp_dir, 'htdemucs', filename)
    return os.path.join(separated_dir, 'vocals.wav'), os.path.join(separated_dir, 'no_vocals.wav')
