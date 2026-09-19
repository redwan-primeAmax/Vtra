from typing import List, Dict
import re

def restore_punctuation_and_sentences(segments: List[Dict]) -> List[Dict]:
    sentences = []
    current_words = []
    current_start = None

    all_words = []
    for seg in segments:
        all_words.extend(seg.get("words", []))

    if not all_words:
        return segments

    for i, word_info in enumerate(all_words):
        if current_start is None:
            current_start = word_info["start"]
        current_words.append(word_info["word"])

        is_last = (i == len(all_words) - 1)
        gap = 0 if is_last else all_words[i+1]["start"] - word_info["end"]
        ends_with_punct = bool(re.search(r'[.!?]$', word_info["word"].strip()))

        # ০.৬ সেকেন্ডের নীরবতা বা পাংচুয়েশন মার্ক থাকলে বাক্য ভাঙা
        if gap > 0.6 or ends_with_punct or is_last:
            text = " ".join(current_words).strip()
            text = re.sub(r'\b(uh|um|ah)\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip()

            if text:
                sentences.append({
                    "start": current_start,
                    "end": word_info["end"],
                    "text": text
                })
            current_words = []
            current_start = None

    return sentences
