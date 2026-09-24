import re
import time
import logging
from typing import List, Dict, Tuple
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests
from dubbing.memory import flush_memory

logger = logging.getLogger("dubbing.translate")

def apply_glossary_placeholders(text: str, glossary: List[str]) -> Tuple[str, Dict[str, str]]:
    """নির্দিষ্ট কীওয়ার্ডকে প্লেসহোল্ডার দিয়ে প্রতিস্থাপন করে যাতে গুগল ট্রান্সলেট তা পরিবর্তন না করে।"""
    mapping = {}
    modified_text = text
    for idx, word in enumerate(glossary):
        placeholder = f"__KEEP_{idx}__"
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        if pattern.search(modified_text):
            modified_text = pattern.sub(placeholder, modified_text)
            mapping[placeholder] = word
    return modified_text, mapping

def restore_glossary_words(translated_text: str, mapping: Dict[str, str]) -> str:
    """অনুবাদ শেষে প্লেসহোল্ডারগুলো সরিয়ে মূল ইংরেজি শব্দ ফিরিয়ে আনে।"""
    result = translated_text
    for placeholder, original_word in mapping.items():
        result = result.replace(placeholder, original_word)
    return result

def translate_sentences_google(sentences: List[Dict], glossary: List[str]) -> List[Dict]:
    """ব্যাচ প্রসেসিং ও রিট্রাই মেকানিজম সহ গুগল ট্রান্সলেট সম্পাদন করে।"""
    if not sentences:
        return []

    translator = GoogleTranslator(source='en', target='bn')
    results = []
    
    # কীওয়ার্ড প্রতিস্থাপন প্রস্তুত করা
    prepared_items = []
    for item in sentences:
        masked_text, mapping = apply_glossary_placeholders(item["text"], glossary)
        prepared_items.append((item, masked_text, mapping))

    # ১৫টি বাক্য করে একসাথে ব্যাচে অনুবাদ
    BATCH_SIZE = 15

    for i in range(0, len(prepared_items), BATCH_SIZE):
        batch = prepared_items[i:i + BATCH_SIZE]
        batch_texts = [b[1] for b in batch]

        translated_batch = None
        max_retries = 5

        # আইপি ব্লকিং এড়াতে রিট্রাই লজিক
        for attempt in range(max_retries):
            try:
                translated_batch = translator.translate_batch(batch_texts)
                time.sleep(0.4) # ব্যাচগুলোর মাঝে নিরাপদ বিরতি
                break
            except TooManyRequests:
                wait_time = (attempt + 1) * 3
                logger.warning(f"গুগল রেট লিমিট দিয়েছে! {wait_time} সেকেন্ড অপেক্ষা করা হচ্ছে...")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"অনুবাদে ত্রুটি ({e}), পুনরায় চেষ্টা করা হচ্ছে...")
                time.sleep(2)

        # যদি ব্যাচ অনুবাদ ব্যর্থ হয়, তবে ফলব্যাক হিসেবে একটি একটি করে চেষ্টা করা
        if not translated_batch or len(translated_batch) != len(batch):
            translated_batch = []
            for item_tuple in batch:
                try:
                    res = translator.translate(item_tuple[1])
                    translated_batch.append(res)
                    time.sleep(0.3)
                except Exception:
                    translated_batch.append(item_tuple[0]["text"])

        # রেজাল্ট সাজানো এবং প্লেসহোল্ডার রিস্টোর
        for idx, (orig_item, _, mapping) in enumerate(batch):
            raw_trans = translated_batch[idx] if idx < len(translated_batch) else orig_item["text"]
            final_bn_text = restore_glossary_words(raw_trans, mapping)

            results.append({
                "start": orig_item["start"],
                "end": orig_item["end"],
                "src_text": orig_item["text"],
                "tgt_text": final_bn_text
            })

    flush_memory()
    return results
