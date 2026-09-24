import re
import time
import random
import logging
from typing import List, Dict, Tuple
import requests
from deep_translator import GoogleTranslator, MyMemoryTranslator
from dubbing.memory import flush_memory

logger = logging.getLogger("dubbing.translate")

# ---------- User-Agent rotation (ব্লক কমাতে সহায়ক) ----------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

def _new_google_translator():
    # কিছু ক্ষেত্রে Google-এর রিজিওনাল ডোমেইন ব্যবহার করলে ব্লক কম হয়
    try:
        return GoogleTranslator(source="en", target="bn")
    except Exception:
        return GoogleTranslator(source="auto", target="bn")

# ---------- Glossary placeholders ----------
def apply_glossary_placeholders(text: str, glossary: List[str]) -> Tuple[str, Dict[str, str]]:
    mapping = {}
    modified_text = text
    for idx, word in enumerate(glossary):
        placeholder = f"KEEPWORD{idx}"
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        if pattern.search(modified_text):
            modified_text = pattern.sub(placeholder, modified_text)
            mapping[placeholder] = word
    return modified_text, mapping

def restore_glossary_words(translated_text: str, mapping: Dict[str, str]) -> str:
    result = translated_text
    for placeholder, original_word in mapping.items():
        result = re.sub(re.escape(placeholder), original_word, result, flags=re.IGNORECASE)
    return result

# ---------- মূল অনুবাদ ফাংশন (Fallback সহ) ----------
def _translate_google(text: str) -> str:
    tr = _new_google_translator()
    return tr.translate(text) or ""

def _translate_mymemory(text: str) -> str:
    tr = MyMemoryTranslator(source="en-US", target="bn-IN")
    return tr.translate(text) or ""

def translate_with_retry(text: str, retries: int = 6) -> str:
    """
    Exponential backoff + jitter সহ অনুবাদ।
    Google ফেইল করলে MyMemory দিয়ে fallback।
    """
    if not text or not text.strip():
        return ""

    last_error = None
    for attempt in range(retries):
        try:
            # মাঝে মাঝে Google রিকোয়েস্ট করা
            result = _translate_google(text)
            if result and result.strip():
                return result
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            # ব্লক ডিটেক্ট
            if any(k in msg for k in ["too many", "429", "blocked", "unusual traffic", "captcha"]):
                wait = min(60, (2 ** attempt) + random.uniform(0, 2))
                logger.warning(f"🚫 Google ব্লক শনাক্ত। {wait:.1f}s অপেক্ষা... ({attempt+1}/{retries})")
            else:
                wait = min(30, (2 ** attempt) + random.uniform(0, 1))
                logger.warning(f"⚠️ অনুবাদ রিট্রাই {attempt+1}/{retries} — {wait:.1f}s")
            time.sleep(wait)

    # Google সব ফেইল → MyMemory দিয়ে চেষ্টা
    try:
        logger.warning("🔁 Google ফেইল, MyMemory fallback ব্যবহার করা হচ্ছে...")
        result = _translate_mymemory(text)
        if result and result.strip():
            return result
    except Exception as e:
        logger.error(f"❌ MyMemory-ও ফেইল: {e}")

    logger.error(f"❌ সব অনুবাদ fallback ব্যর্থ। মূল টেক্সট রাখা হচ্ছে। শেষ এরর: {last_error}")
    return text

# ---------- Character-based chunking ----------
def _chunk_by_chars(items: List[Tuple], max_chars: int = 1500, separator: str = " ||| "):
    """
    বাক্যগুলোকে ক্যারেক্টার-লিমিট অনুযায়ী ভাগ করা (Google-এর নিরাপদ সীমা)।
    """
    chunks, current, current_len = [], [], 0
    for it in items:
        text = it[1]  # masked text
        add_len = len(text) + len(separator)
        if current and current_len + add_len > max_chars:
            chunks.append(current)
            current, current_len = [], 0
        current.append(it)
        current_len += add_len
    if current:
        chunks.append(current)
    return chunks

def translate_sentences_google(sentences: List[Dict], glossary: List[str]) -> List[Dict]:
    if not sentences:
        return []

    # ১) Glossary placeholder প্রয়োগ
    prepared = []
    for item in sentences:
        masked, mapping = apply_glossary_placeholders(item["text"], glossary)
        prepared.append((item, masked, mapping))

    # ২) ক্যারেক্টার-ভিত্তিক নিরাপদ চাঙ্কিং
    chunk_groups = _chunk_by_chars(prepared, max_chars=1500)
    logger.info(f"🌐 মোট {len(sentences)} বাক্য → {len(chunk_groups)} টি চাঙ্কে অনুবাদ হবে")

    results = []
    for ci, chunk in enumerate(chunk_groups, 1):
        separator = " ||| "
        combined = separator.join([c[1] for c in chunk])

        translated_combined = translate_with_retry(combined)

        # ৩) সেপারেটর দিয়ে ভাগ করার চেষ্টা (একাধিক ভ্যারিয়েন্ট হ্যান্ডেল)
        parts = None
        for sep_variant in [" ||| ", "|||", " | | | ", "\n\n"]:
            if sep_variant in translated_combined:
                candidate = [p.strip() for p in translated_combined.split(sep_variant)]
                if len(candidate) == len(chunk):
                    parts = candidate
                    break

        # ৪) ভাগ করতে ব্যর্থ হলে প্রতিটি বাক্য আলাদা করে অনুবাদ (small batch, safe)
        if parts is None or len(parts) != len(chunk):
            logger.warning(f"⚠️ চাঙ্ক {ci}: separator mismatch, আলাদা করে অনুবাদ করা হচ্ছে...")
            parts = []
            for c in chunk:
                parts.append(translate_with_retry(c[1]))
                time.sleep(random.uniform(0.8, 1.8))

        # ৫) Glossary restore
        for idx, (orig_item, _, mapping) in enumerate(chunk):
            raw_trans = parts[idx] if idx < len(parts) else orig_item["text"]
            final_bn = restore_glossary_words(raw_trans, mapping)
            results.append({
                "start": orig_item["start"],
                "end": orig_item["end"],
                "src_text": orig_item["text"],
                "tgt_text": final_bn,
            })

        # ৬) প্রতি চাঙ্কের পর এলোমেলো বিরতি (বট ডিটেকশন এড়াতে)
        time.sleep(random.uniform(2.5, 4.5))
        logger.info(f"✅ চাঙ্ক {ci}/{len(chunk_groups)} সম্পন্ন")

    flush_memory()
    return results