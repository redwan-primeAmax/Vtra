from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dubbing.memory import flush_memory

def translate_sentences(
    sentences: List[Dict], 
    model_name: str = "facebook/nllb-200-distilled-600M",
    src_lang: str = "eng_Latn",
    tgt_lang: str = "ben_Beng"
) -> List[Dict]:
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    translated = []
    for item in sentences:
        text = item["text"]
        inputs = tokenizer(text, return_tensors="pt").to(device)

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=256
        )
        res_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        translated.append({
            "start": item["start"],
            "end": item["end"],
            "src_text": text,
            "tgt_text": res_text
        })

    del model
    del tokenizer
    flush_memory()
    return translated
