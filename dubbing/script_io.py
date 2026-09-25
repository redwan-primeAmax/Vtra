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
    for ext in VIDEO_EXTS:
        files = sorted(folder.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def find_translated_file(folder: Path) -> Optional[Path]:
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
    video_stem: str = "video",
) -> str:
    json_filename = f"{video_stem}.translated.json"
    total_blocks = len(segments)

    lines = [
        f"# Translation Task: {src_lang} → {tgt_lang}",
        "",
        "## 🚨 READ FIRST — OUTPUT RULES (STRICT)",
        "",
        f"**১. আউটপুট ফাইলের নাম:** `{json_filename}`",
        f"**২. মোট ব্লক সংখ্যা:** {total_blocks}",
        "**৩. আউটপুট ফরম্যাট:** শুধুমাত্র একটি `json` কোড-ব্লক।",
        "   - কোনো ভূমিকা/ব্যাখ্যা/শুভেচ্ছা লিখবেন না।",
        "   - কোড-ব্লকের বাইরে একটি অক্ষরও লিখবেন না।",
        "   - প্রতিটি ব্লকের জন্য `segments` অ্যারেতে ঠিক একটি entry দিবেন।",
        f"   - entry সংখ্যা অবশ্যই {total_blocks} হতে হবে — কম/বেশি নয়।",
        "   - `start` এবং `end` ইনপুটের সাথে **হুবহু** মিলবে (নম্বর বদলাবেন না)।",
        "",
        "**৪. JSON schema (হুবহু এই কী-গুলো ব্যবহার করুন):**",
        "```json",
        "{",
        '  "_type": "translated",',
        f'  "_src_lang": "{src_lang}",',
        f'  "_tgt_lang": "{tgt_lang}",',
        '  "segments": [',
        '    {"start": 0.00, "end": 2.50, "text": "বাংলা অনুবাদ"},',
        '    {"start": 2.50, "end": 5.10, "text": "পরের লাইন"},',
        "    ...",
        "  ]",
        "}",
        "```",
        "",
        "**৫. অনুবাদের নিয়ম:**",
        "- `max_chars` এর মধ্যে থাকুন (৫ সেকেন্ডে পড়ার সীমা)",
        "- Glossary-র term ইংরেজিতেই রাখুন",
        "- বাংলা সংখ্যা শব্দে লিখুন (১২ → \"বারো\")",
        "- সংলাপ হলে স্বাভাবিক কথ্য বাংলা, বর্ণনা হলে সহজ বাংলা",
        "- যুক্তাক্ষর স্বাভাবিক, বাংলা একাডেমি বানান রীতি",
        "",
        "## Glossary (keep as-is in English)",
        ", ".join(glossary) if glossary else "(none)",
        "",
        "---",
        "",
        f"## 📋 Blocks to Translate ({total_blocks}টি)",
        "",
    ]

    for i, seg in enumerate(segments, 1):
        dur = max(seg["end"] - seg["start"], 0.1)
        max_chars = max(int(dur * _CHARS_PER_SEC), 10)
        lines.append(f"### Block {i} / {total_blocks}")
        lines.append(f"- start: {seg['start']:.3f}")
        lines.append(f"- end: {seg['end']:.3f}")
        lines.append(f"- duration: {dur:.2f} s")
        lines.append(f"- max_chars: {max_chars}")
        lines.append(f"- src: \"{seg['text'].strip()}\"")
        lines.append("")

    lines += [
        "---",
        "",
        "## ✅ NOW — FINAL OUTPUT",
        "",
        "উপরের সবগুলো Block অনুবাদ করে **একটি মাত্র JSON কোড-ব্লক** নিচে দিন।",
        f"এবং ফাইলটি সেভ করুন এই নামে: **`{json_filename}`**",
        "",
        "```json",
        "{",
        '  "_type": "translated",',
        f'  "_src_lang": "{src_lang}",',
        f'  "_tgt_lang": "{tgt_lang}",',
        '  "segments": [',
        '    {"start": 0.000, "end": 0.000, "text": "..."}',
        f"    // মোট {total_blocks}টি entry — কোনোটি বাদ যাবে না",
        "  ]",
        "}",
        "```",
        "",
        "⚠️ মনে রাখবেন: সমস্ত `...` বাদ দিয়ে বাস্তব অনুবাদ বসাতে হবে।",
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
        build_ai_markdown(segments, glossary, video_stem=video_stem),
        encoding="utf-8",
    )
    logger.info(f"📝 AI markdown সেভ: {out.name}")
    return out


# ---------- AI রেসপন্স পার্স ----------
_BLOCK_PATTERN = re.compile(
    r'^\s*tgt\s*:\s*["“”](.+?)["“”]\s*$',
    re.MULTILINE,
)


def parse_ai_response(response_text: str) -> List[str]:
    """
    AI-এর উত্তর থেকে translation বের করে।
    প্রায়োরিটি: JSON (কোড-ব্লকে বা raw) → পুরনো tgt: "..." ফরম্যাট (fallback)।
    """
    text = response_text.strip()

    # ১) ```json ... ``` কোড-ব্লক
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict) and "segments" in data:
                return [str(s.get("text", "")).strip() for s in data["segments"]]
        except Exception:
            pass

    # ২) raw JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "segments" in data:
            return [str(s.get("text", "")).strip() for s in data["segments"]]
    except Exception:
        pass

    # ৩) কোড-ব্লক ছাড়া { ... } অংশ খুঁজে বের করা
    m2 = re.search(r"\{.*\}", text, re.DOTALL)
    if m2:
        try:
            data = json.loads(m2.group(0))
            if isinstance(data, dict) and "segments" in data:
                return [str(s.get("text", "")).strip() for s in data["segments"]]
        except Exception:
            pass

    # ৪) Fallback: পুরনো `tgt: "..."` প্যাটার্ন
    return [mm.strip() for mm in _BLOCK_PATTERN.findall(text)]