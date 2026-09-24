import subprocess
from pathlib import Path
from typing import List, Dict
import soundfile as sf
from dubbing.memory import flush_memory

def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return float(res.stdout.strip())

def get_audio_duration(wav_path: Path) -> float:
    data, sr = sf.read(str(wav_path))
    return len(data) / float(sr)

def sync_and_assemble_video(video_path: Path, bgm_wav: Path, items: List[Dict], output_dir: Path, output_video: Path, bgm_volume: float = 0.3):
    total_duration = get_video_duration(video_path)
    blocks = []
    current_time = 0.0

    for item in items:
        v_start = item["start"]
        v_end = item["end"]

        if v_start > current_time + 0.05:
            blocks.append({"v_start": current_time, "v_end": v_start, "tts_wav": None, "is_gap": True})

        blocks.append({"v_start": v_start, "v_end": max(v_end, v_start + 0.1), "tts_wav": Path(item["tts_wav"]), "is_gap": False})
        current_time = max(v_end, current_time)

    if current_time < total_duration - 0.05:
        blocks.append({"v_start": current_time, "v_end": total_duration, "tts_wav": None, "is_gap": True})

    concat_list_path = output_dir / "concat_list.txt"

    with open(concat_list_path, "w", encoding="utf-8") as f_concat:
        for idx, block in enumerate(blocks):
            v_start, v_end = block["v_start"], block["v_end"]
            v_dur = max(v_end - v_start, 0.1)
            seg_video_path = output_dir / f"block_{idx:04d}.mp4"

            if block["is_gap"]:
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-ss", f"{v_start:.3f}", "-to", f"{v_end:.3f}",
                    "-i", str(video_path), "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", str(seg_video_path)
                ]
            else:
                tts_wav = block["tts_wav"]
                audio_dur = get_audio_duration(tts_wav)
                pts_ratio = audio_dur / v_dur

                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-ss", f"{v_start:.3f}", "-to", f"{v_end:.3f}",
                    "-i", str(video_path), "-i", str(tts_wav),
                    "-filter_complex", f"[0:v]setpts={pts_ratio:.4f}*PTS[v]",
                    "-map", "[v]", "-map", "1:a:0", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-b:a", "192k", str(seg_video_path)
                ]

            subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            f_concat.write(f"file '{seg_video_path.resolve()}'\n")

    temp_speech_video = output_dir / "temp_speech_video.mp4"
    concat_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list_path), "-c", "copy", str(temp_speech_video)]
    subprocess.run(concat_cmd, capture_output=True, text=True)

    final_mix_cmd = [
        "ffmpeg", "-y", "-i", str(temp_speech_video), "-i", str(bgm_wav),
        "-filter_complex", f"[1:a]volume={bgm_volume}[bgm];[0:a][bgm]amix=inputs=2:duration=first[a]",
        "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(output_video)
    ]

    res = subprocess.run(final_mix_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        temp_speech_video.replace(output_video)

    flush_memory()
