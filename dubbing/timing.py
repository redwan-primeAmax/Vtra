from typing import List, Dict

def adapt_timing_budget(translated_items: List[Dict]) -> List[Dict]:
    adapted = []
    for item in translated_items:
        duration = item["end"] - item["start"]
        tgt_text = item["tgt_text"]

        # বাংলায় গড়ে প্রতি সেকেন্ডে ১৪টি অক্ষরের উচ্চারণ গতি ধরে গতি হিসাব
        char_count = len(tgt_text)
        est_duration = char_count / 14.0

        speed_ratio = 1.0
        if est_duration > duration and duration > 0:
            speed_ratio = min(1.25, est_duration / duration)

        adapted.append({
            **item,
            "duration": duration,
            "target_speed": round(speed_ratio, 2)
        })
    return adapted