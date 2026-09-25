"""
স্ক্রিপ্ট I/O — transcribed.json, translated.json, for_ai.md
ফোল্ডার স্ক্যানিং ও AI markdown জেনারেশন।
"""
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger("dubbing.script_io")

TRANSCRIBED_SUFFIX = ".transcribed.json"
TRANSLATED_SUFFIX = ".translated.json"
AI_MD_SUFFIX = ".for_ai.md"

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi")


# ---------- স্ক্যানিং ----------
def find_video_file(folder: Path) -> Optional[Path]:
    """ফোল্ডারের প্রথম ভিডিও ফাইল খুঁজে বের করে।"""
    for ext in VIDEO_EXTS:
        files = sorted(folder.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def find_translated_file(folder: Path) -> Optional[Path]:
    """ফোল্ডারে *.translated.json খুঁজে বের করে।"""
    files = sorted(folder.glob(f"*{TRANSLATED_SUFFIX}"))
    return files[0] if files else None


# ---------- JSON লোড/সেভ ----------
def _dump_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_transcribed_json(
    folder: Path,
    video_stem: str,
    segments: List[Dict],
    src_lang: str = "en",
    model: str = "whisper-large-v3",
) -> Path:
    """Whisper-এর আউটপুট transcribed.json হিসাবে সেভ করে।"""
    out = folder / f"{video_stem}{TRANSCRIBED_SUFFIX}"
    payload = {
        "_type": "transcribed",
        "_src_lang": src_lang,
        "_model": model,
        "_created": datetime.now().isoformat(timespec="seconds"),
        "segments": [
            {
                "start": round(float(s["start"]), 3),
                "end": round(float(s["end"]), 3),
                "text": str(s["text"]).strip(),
            }
            for s in segments
        ],
    }
    _dump_json(out, payload)
    logger.info(f"💾 transcribed.json সেভ: {out.name}")
    return out


def load_translated_segments(path: Path) -> List[Dict]:
    """translated.json থেকে segments লোড করে (বাংলা টেক্সট সহ)।"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for s in data.get("segments", []):
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        out.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": text,
        })
    return out


# ---------- AI Markdown ----------
_CHARS_PER_SEC = 14  # বাংলায় গড়ে ১৪ অক্ষর/সেকেন্ড


def build_ai_markdown(
    segments: List[Dict],
    glossary: List[str],
    src_lang: str = "en",
    tgt_lang: str = "bn",
) -> str:
    lines = [
        f"# Translation Task: {src_lang} → {tgt_lang}",
        "",
        "## Instructions",
        "- প্রতিটি ব্লক আলাদা অনুবাদ করুন",
        "- `max_chars` এর মধ্যে থাকুন (৫ সেকেন্ডে পড়ার সীমা)",
        "- `keep` লিস্টে থাকা শব্দ ইংরেজিতেই রাখুন",
        "- বাংলা সংখ্যা শব্দে লিখুন (১২ → \"বারো\")",
        "- সংলাপ হলে স্বাভাবিক কথ্য বাংলা, বর্ণনা হলে সহজ বাংলা",
        "- যুক্তাক্ষর স্বাভাবিক রাখুন, বানান বাংলা একাডেমি রীতি অনুসারে",
        "",
        "## Glossary (keep as-is)",
        ", ".join(glossary),
        "",
        "---",
        "",
    ]

    for i, seg in enumerate(segments, 1):
        dur = max(seg["end"] - seg["start"], 0.1)
        max_chars = max(int(dur * _CHARS_PER_SEC), 10)
        lines.append(f"## Block {i}")
        lines.append(f"- time: {seg['start']:.2f} – {seg['end']:.2f} s")
        lines.append(f"- duration: {dur:.2f} s")
        lines.append(f"- max_chars: {max_chars}")
        lines.append(f"- src: \"{seg['text'].strip()}\"")
        lines.append("")

    lines += [
        "---",
        "",
        "## Expected Output Format",
        "প্রতিটি ব্লকের জন্য শুধু এই ফরম্যাটে উত্তর দিন:",
        "",
        "### Block 1",
        "tgt: \"বাংলা অনুবাদ\"",
        "",
        "### Block 2",
        "tgt: \"বাংলা অনুবাদ\"",
        "",
    ]
    return "\n".join(lines)


def save_ai_markdown(
    folder: Path,
    video_stem: str,
    segments: List[Dict],
    glossary: List[str],
) -> Path:
    out = folder / f"{video_stem}{AI_MD_SUFFIX}"
    out.write_text(
        build_ai_markdown(segments, glossary),
        encoding="utf-8",
    )
    logger.info(f"📝 AI markdown সেভ: {out.name}")
    return out


# ---------- AI রেসপন্স পার্স ----------
_AI_PATTERN = re.compile(
    r'^\s*tgt\s*:\s*["“”](.+?)["“”]\s*$',
    re.MULTILINE,
)


def parse_ai_response(response_text: str) -> List[str]:
    """AI-এর উত্তল থেকে tgt: \"...\" গুলো ক্রমানুসারে বের করে।"""
    return [m.strip() for m in _AI_PATTERN.findall(response_text)]