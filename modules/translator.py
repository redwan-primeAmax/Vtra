import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config.settings import Config
from utils.gpu import clear_vram
from utils.logger import logger

def _split_text_by_words(text: str, max_words: int = 25) -> list[str]:
    """যদি কোনো সেগমেন্টে পাংচুয়েশন না থাকে, তবে ২৫ শব্দ পর পর স্প্লিট করার ফাংশন"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks

def translate_en_to_bn(segments: list[dict]) -> list[dict]:
    if not segments:
        return []

    logger.info("NLLB-200 অনুবাদ মডেল লোড করা হচ্ছে...")
    tokenizer = AutoTokenizer.from_pretrained(Config.TRANSLATOR_MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(Config.TRANSLATOR_MODEL).to(Config.DEVICE)

    translated_segments = []

    for idx, seg in enumerate(segments):
        english_text = seg.get("text", "").strip()
        
        if not english_text:
            seg["translated_text"] = ""
            translated_segments.append(seg)
            continue

        try:
            # যদি বাক্য বড় হয় তবে ছোট টুকরো করে অনুবাদ করা হবে
            text_chunks = _split_text_by_words(english_text, max_words=25)
            bn_chunks = []

            for chunk in text_chunks:
                inputs = tokenizer(chunk, return_tensors="pt").to(Config.DEVICE)
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id["ben_Beng"],
                    max_length=256
                )
                bn_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                bn_chunks.append(bn_text.strip())

            full_bn_text = " ".join(bn_chunks)
            seg_copy = dict(seg)
            seg_copy["translated_text"] = full_bn_text
            translated_segments.append(seg_copy)
            
        except Exception as e:
            logger.error(f"অনুবাদে সমস্যা (সেগমেন্ট {idx}): {e}")
            seg_copy = dict(seg)
            seg_copy["translated_text"] = english_text
            translated_segments.append(seg_copy)

    del model
    del tokenizer
    clear_vram()
    return translated_segments

def save_translation_to_txt(segments: list[dict], output_path: str):
    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("   FULL DUBBING ANALYSIS (ENGLISH -> BENGALI)     \n")
            f.write("==================================================\n\n")
            
            for idx, seg in enumerate(segments, 1):
                start = seg.get('start', 0.0)
                end = seg.get('end', 0.0)
                duration = end - start
                en_text = seg.get('text', '')
                bn_text = seg.get('translated_text', '')
                
                f.write(f"[{idx:03d}] TIME: {start:.2f}s -> {end:.2f}s | DURATION: {duration:.2f}s\n")
                f.write(f"EN : {en_text}\n")
                f.write(f"BN : {bn_text}\n")
                f.write("-" * 60 + "\n")
                
        logger.info(f"অনুবাদ সহ পূর্ণাঙ্গ এনালাইসিস ফাইল সেভ হয়েছে: {output_path}")
    except Exception as e:
        logger.error(f"অনুবাদ ফাইল সেভ করতে সমস্যা হয়েছে: {e}")
