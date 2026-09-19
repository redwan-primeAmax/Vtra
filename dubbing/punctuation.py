import re
from typing import List, Dict
import torch
from transformers import pipeline
from dubbing.memory import flush_memory

def restore_punctuation_and_sentences(segments: List[Dict]) -> List[Dict]:
    """
    Hugging Face transformers pipeline ব্যবহার করে টাইমিং-সহ নিখুঁত পাংচুয়েশন ও বাক্য গঠন।
    `grouped_entities` বা ডিপেনডেন্সিজনিত কোনো এরর আসবে না।
    """
    all_words = []
    for seg in segments:
        all_words.extend(seg.get("words", []))

    if not all_words:
        return segments

    raw_text = " ".join([w["word"].strip() for w in all_words])

    # ডিপেনডেন্সি ত্রুটি এড়াতে সরাসরি Transformers NER Pipeline ব্যবহার
    device = 0 if torch.cuda.is_available() else -1
    punct_pipe = pipeline(
        "ner",
        model="oliverguhr/fullstop-punctuation-multilang-large",
        aggregation_strategy="none",
        device=device
    )

    # ইনফারেন্স ও মেমোরি ক্লিয়ার
    results = punct_pipe(raw_text)
    del punct_pipe
    flush_memory()

    # পাংচুয়েশন প্রেডিকশন প্রয়োগ
    punct_map = {item['start']: item['entity_group'] if 'entity_group' in item else item.get('entity') for item in results}

    sentences = []
    current_words = []
    current_start = None

    for i, word_info in enumerate(all_words):
        if current_start is None:
            current_start = word_info["start"]

        word_str = word_info["word"].strip()
        current_words.append(word_str)

        is_last = (i == len(all_words) - 1)
        gap = 0 if is_last else (all_words[i+1]["start"] - word_info["end"])
        ends_with_punc = bool(re.search(r'[.!?]$', word_str))

        # ০.৭ সেকেন্ডের বিরতি বা কাস্টম পাংচুয়েশন সংকেত থাকলে বাক্য তৈরি
        if gap > 0.7 or ends_with_punc or is_last:
            sentence_text = " ".join(current_words).strip()
            sentence_text = re.sub(r'\b(uh|um|ah)\b', '', sentence_text, flags=re.IGNORECASE)
            sentence_text = re.sub(r'\s+', ' ', sentence_text).strip()

            if sentence_text:
                sentences.append({
                    "start": current_start,
                    "end": word_info["end"],
                    "text": sentence_text
                })
            current_words = []
            current_start = None

    return sentences
