import re
import logging
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dubbing.memory import flush_memory

logger = logging.getLogger("dubbing.translate")

# অফলাইন মডেল ও টোকেনাইজার লোড
MODEL_NAME = "Helsinki-NLP/opus-mt-en-bengali"
tokenizer = None
model = None

def get_translation_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    return tokenizer, model

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

def translate_sentences_google(sentences: List[Dict], glossary: List[str]) -> List[Dict]:
    if not sentences:
        return []

    tok, md = get_translation_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    prepared_items = []
    for item in sentences:
        masked_text, mapping = apply_glossary_placeholders(item["text"], glossary)
        prepared_items.append((item, masked_text, mapping))

    BATCH_SIZE = 16
    for i in range(0, len(prepared_items), BATCH_SIZE):
        batch = prepared_items[i:i + BATCH_SIZE]
        batch_texts = [b[1] for b in batch]

        inputs = tok(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated_tokens = md.generate(**inputs, max_length=512)
        
        translated_batch = tok.batch_decode(translated_tokens, skip_special_tokens=True)

        for idx, (orig_item, _, mapping) in enumerate(batch):
            raw_trans = translated_batch[idx]
            final_bn_text = restore_glossary_words(raw_trans, mapping)

            results.append({
                "start": orig_item["start"],
                "end": orig_item["end"],
                "src_text": orig_item["text"],
                "tgt_text": final_bn_text
            })

    flush_memory()
    return results
