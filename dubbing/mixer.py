import subprocess
import json
from pathlib import Path
from typing import List, Dict
import soundfile as sf
import numpy as np

def time_fit_and_mix(items: List[Dict], total_duration: float, output_mixed_wav: Path):
    target_sr = 24000
    canvas = np.zeros(int(total_duration * target_sr), dtype=np.float32)

    for item in items:
        wav_path = Path(item["tts_wav"])
        start_sec = item["start"]
        data, sr = sf.read(str(wav_path))

        if len(data.shape) > 1:
            data = data.mean(axis=1)

        if sr != target_sr:
            num_samples = int(len(data) * target_sr / sr)
            data = np.interp(np.linspace(0, len(data), num_samples), np.arange(len(data)), data)

        start_idx = int(start_sec * target_sr)
        end_idx = min(start_idx + len(data), len(canvas))
        clip_len = end_idx - start_idx

        if clip_len > 0:
            canvas[start_idx:end_idx] += data[:clip_len]

    max_val = np.max(np.abs(canvas))
    if max_val > 0:
        canvas = canvas / max_val * 0.95

    sf.write(str(output_mixed_wav), canvas, target_sr)

def get_video_codec(video_path: Path) -> str:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip().lower()

def assemble_final_video(video_path: Path, audio_wav: Path, output_video: Path):
    """
    ভিডিও কোডেক চেক করে সিদ্ধান্ত নেয় স্ট্রিম কপি করবে নাকি T4 GPU দিয়ে এনকোড করবে।
    .webm (VP8/VP9) ফাইলগুলোকে GPU Hardware Acceleration দিয়ে H.264 এ কনভার্ট করে।
    """
    codec = get_video_codec(video_path)
    ext = video_path.suffix.lower()

    # যদি ফাইল WebM হয় বা কোডেক VP8/VP9 হয় তবে GPU দিয়ে H.264 এনকোড করবে
    needs_reencode = ext == ".webm" or codec in ["vp8", "vp9", "theora", "av1"]

    if needs_reencode:
        vcodec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"] # Colab T4 GPU এনকোডার
    else:
        vcodec_args = ["-c:v", "copy"]

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_wav),
        *vcodec_args,
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_video)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # NVENC কোনো কারণে ফেল করলে CPU ফলব্যাক (libx264)
    if res.returncode != 0 and needs_reencode:
        cmd[cmd.index("h264_nvenc")] = "libx264"
        res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg ভিডিও অ্যাসেম্বলি ব্যর্থ: {res.stderr}")
