import torch
import logging
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dubbing.memory import flush_memory

logger = logging.getLogger("dubbing.translate")

def translate_sentences_local(
    sentences: List[Dict], 
    model_name: str = "facebook/nllb-200-1.3B",
    src_lang: str = "eng_Latn",
    tgt_lang: str = "ben_Beng",
    batch_size: int = 16
) -> List[Dict]:
    """
    লোকাল NLLB-200-1.3B মডেল এবং ব্যাচ ইনফারেন্স ব্যবহার করে অতি দ্রুত ও নির্ভুল বাংলা অনুবাদ করে।
    """
    if not sentences:
        return []

    logger.info(f"লোকাল অনুবাদ মডেল লোড হচ্ছে: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    texts = [s["text"] for s in sentences]
    translated_texts = []
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    # ব্যাচ প্রসেসিং: একসাথে একাধিক বাক্য GPU-তে অনুবাদ হবে (অত্যন্ত দ্রুত)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=256,
                num_beams=2,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated_texts.extend(decoded)

    results = []
    for item, tgt in zip(sentences, translated_texts):
        results.append({
            "start": item["start"],
            "end": item["end"],
            "src_text": item["text"],
            "tgt_text": tgt
        })

    del model
    del tokenizer
    flush_memory()
    return results
