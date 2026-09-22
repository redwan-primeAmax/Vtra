import json
import logging
import re
from typing import List, Dict
import requests
from dubbing.memory import flush_memory

logger = logging.getLogger("dubbing.translate")

def translate_sentences_api(sentences: List[Dict], api_key: str) -> List[Dict]:
    if not api_key:
        raise ValueError("API Key পাওয়া যায়নি! অনুগ্রহ করে সঠিক Gemini API Key সরবরাহ করুন।")

    if not sentences:
        return []

    headers = {"Content-Type": "application/json"}
    
    input_data = [
        {
            "id": idx, 
            "text": item["text"], 
            "allowed_seconds": item.get("duration", 4.0)
        } 
        for idx, item in enumerate(sentences)
    ]

    prompt = f"""
You are a professional video dubbing translator. Translate the English transcript into natural, conversational, spoken Bengali (চলিত বাংলা) suitable for voiceover.

CRITICAL DURATION RULE:
The translated Bengali voice will be spoken within 'allowed_seconds'. Keep your translation concise, clear, and natural so that it fits smoothly within that time without sounding rushed.

Return ONLY a valid JSON array of objects with "id" and "tgt_text".

Input Segments:
{json.dumps(input_data, ensure_ascii=False)}
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    res = requests.post(url, json=payload, headers=headers)
    if res.status_code != 200:
        raise RuntimeError(f"Translation API error ({res.status_code}): {res.text}")

    try:
        response_data = res.json()
        raw_json = response_data["candidates"][0]["content"]["parts"][0]["text"]
        
        json_match = re.search(r'\[.*\]', raw_json, re.DOTALL)
        if json_match:
            raw_json = json_match.group(0)

        translated_list = json.loads(raw_json)
        translated_map = {item["id"]: item["tgt_text"] for item in translated_list}
    except Exception as e:
        logger.error(f"API প্রতিক্রিয়া প্রসেস করতে ব্যর্থ: {e}")
        raise RuntimeError(f"API এর প্রতিক্রিয়া প্রসেস করতে ব্যর্থ: {e}")

    results = []
    for idx, item in enumerate(sentences):
        results.append({
            "start": item["start"],
            "end": item["end"],
            "src_text": item["text"],
            "tgt_text": translated_map.get(idx, item["text"])
        })

    flush_memory()
    return results
