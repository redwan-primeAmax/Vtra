import re
from typing import List, Dict

def restore_punctuation_and_sentences(segments: List[Dict], max_duration: float = 5.0) -> List[Dict]:
    all_words = []
    for seg in segments:
        all_words.extend(seg.get("words", []))

    if not all_words:
        results = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                results.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text
                })
        return results

    sentences = []
    current_words = []
    current_start = None

    for i, word_info in enumerate(all_words):
        if current_start is None:
            current_start = word_info["start"]

        word_str = word_info["word"].strip()
        current_words.append(word_str)

        current_duration = word_info["end"] - current_start
        is_last = (i == len(all_words) - 1)
        ends_with_punc = bool(re.search(r'[.!?]$', word_str))

        if ends_with_punc or current_duration >= max_duration or is_last:
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
