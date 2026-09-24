import re
import time
import logging
from typing import List, Dict, Tuple
from deep_translator import GoogleTranslator
from dubbing.memory import flush_memory

logger = logging.getLogger("dubbing.translate")

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

def translate_with_retry(text: str, retries: int = 5) -> str:
    """ব্লক এড়াতে রিট্রাই লজিকসহ গুগল ট্রান্সলেটর।"""
    translator = GoogleTranslator(source='auto', target='bn')
    for attempt in range(retries):
        try:
            translated = translator.translate(text)
            if translated:
                return translated
        except Exception as e:
            wait_time = (attempt + 1) * 2  # ১ম বার ২ সে, ২য় বার ৪ সে, ৩য় বার ৬ সে...
            logger.warning(f"অনুবাদে বিলম্ব/ব্লক শনাক্ত হয়েছে। {wait_time} সেকেন্ড পর পুনরায় চেষ্টা করা হচ্ছে... ({attempt+1}/{retries})")
            time.sleep(wait_time)
            
    # সব রিট্রাই ব্যর্থ হলে মূল টেক্সটই ফেরত দেবে যেন পাইপলাইন ক্র্যাশ না করে
    return text

def translate_sentences_google(sentences: List[Dict], glossary: List[str]) -> List[Dict]:
    if not sentences:
        return []

    results = []
    prepared_items = []

    for item in sentences:
        masked_text, mapping = apply_glossary_placeholders(item["text"], glossary)
        prepared_items.append((item, masked_text, mapping))

    # বার বার আলাদা রিকোয়েস্ট না পাঠিয়ে ২০০টি করে বাক্য একটি সিঙ্গেল স্ট্রিং বানিয়ে একবারে অনুবাদ করা (স্মার্ট চ্যঙ্কিং)
    CHUNK_SIZE = 50
    for i in range(0, len(prepared_items), CHUNK_SIZE):
        chunk = prepared_items[i:i + CHUNK_SIZE]
        
        # সেপারেটর দিয়ে একসাথে জোড়া লাগানো
        combined_text = " ||| ".join([item[1] for item in chunk])
        
        # ট্রান্সলেট রিকোয়েস্ট
        translated_combined = translate_with_retry(combined_text)
        
        # সেপারেটর দিয়ে অনুবাদকৃত বাক্যগুলোকে আলাদা করা
        translated_parts = [p.strip() for p in translated_combined.split("|||")]

        # যদি কোনো কারণে বিভক্ত সংখ্যার গরমিল হয়, তবে প্রতিটি আলাদা করে অনুবাদ করা হবে
        if len(translated_parts) != len(chunk):
            translated_parts = [translate_with_retry(item[1]) for item in chunk]

        for idx, (orig_item, _, mapping) in enumerate(chunk):
            raw_trans = translated_parts[idx] if idx < len(translated_parts) else orig_item["text"]
            final_bn_text = restore_glossary_words(raw_trans, mapping)

            results.append({
                "start": orig_item["start"],
                "end": orig_item["end"],
                "src_text": orig_item["text"],
                "tgt_text": final_bn_text
            })
            
        time.sleep(1) # আইপি ব্লক এড়াতে প্রতি ব্যাচের পর ১ সেকেন্ড বিরতি

    flush_memory()
    return results
