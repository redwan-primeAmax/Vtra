import re
import time
import logging
from typing import List, Dict, Tuple
from deep_translator import GoogleTranslator
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
    """গুগল ট্রান্সলেট ব্যবহার করে দ্রুত অনুবাদ সম্পাদন করে।"""
    if not sentences:
        return []

    translator = GoogleTranslator(source='en', target='bn')
    results = []

    for item in sentences:
        src_text = item["text"]
        
        # ১. গ্লসারি প্রসেসিং
        masked_text, mapping = apply_glossary_placeholders(src_text, glossary)
        
        # ২. অনুবাদ এবং আইপি ব্লক এড়াতে ক্ষুদ্র বিলম্ব
        try:
            translated = translator.translate(masked_text)
            time.sleep(0.05)  # ৫০ মিলি-সেকেন্ডের সেফটি ডিলে
        except Exception as e:
            logger.warning(f"গুগল ট্রান্সলেট ব্যাকঅফ, পুনরায় চেষ্টা করা হচ্ছে: {e}")
            time.sleep(1.0)
            translated = translator.translate(masked_text)

        # ৩. প্লেসহোল্ডার রিস্টোর
        final_bn_text = restore_glossary_words(translated, mapping)

        results.append({
            "start": item["start"],
            "end": item["end"],
            "src_text": src_text,
            "tgt_text": final_bn_text
        })

    flush_memory()
    return results
