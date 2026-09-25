"""
IndicTrans2 (en-indic distilled 200M) দিয়ে লোকাল অনুবাদ।
Google Translate-এর কোনো API/IP ব্যবহার হয় না — সম্পূর্ণ অফলাইন।
MIT License — Bangladesh থেকে ব্যবহার করা সম্পূর্ণ বৈধ।
"""
import re
import logging
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from dubbing.memory import flush_memory

logger = logging.getLogger("dubbing.translate")

# ---------- ভাষা কোড ----------
SRC_LANG = "eng_Latn"
TGT_LANG = "ben_Beng"

# ---------- মডেল (ungated MIT mirror) ----------
MODEL_NAME = "naklitechie/indictrans2-en-indic-dist-200M"

# ---------- সিঙ্গেলটন (মডেল শুধু একবার লোড হবে) ----------
_model: Optional[AutoModelForSeq2SeqLM] = None
_tokenizer: Optional[AutoTokenizer] = None
_processor: Optional[IndicProcessor] = None


def _load_model(device: str = "cuda"):
    """একবারই মডেল লোড হবে (singleton)। দ্বিতীয়বার কল করলে ক্যাশ থেকে দেবে।"""
    global _model, _tokenizer, _processor
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer, _processor

    logger.info(f"📥 IndicTrans2 (200M) লোড হচ্ছে (device={device})...")

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,   # GPU-তে FP16
        attn_implementation="flash_attention_2",
    ).to(device)
    _model.eval()

    _processor = IndicProcessor(inference=True)

    logger.info("✅ IndicTrans2 মডেল প্রস্তুত")
    return _model, _tokenizer, _processor


# ---------- Glossary মস্কিং (Messi, Ronaldo ইত্যাদি অপরিবর্তিত রাখতে) ----------
def apply_glossary_placeholders(text: str, glossary: List[str]) -> Tuple[str, Dict[str, str]]:
    mapping = {}
    modified = text
    for idx, word in enumerate(glossary):
        placeholder = f"ZZKEEP{idx}ZZ"
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        if pattern.search(modified):
            modified = pattern.sub(placeholder, modified)
            mapping[placeholder] = word
    return modified, mapping


def restore_glossary_words(text: str, mapping: Dict[str, str]) -> str:
    result = text
    for placeholder, original in mapping.items():
        pattern = re.compile(r"\s*".join(re.escape(c) for c in placeholder), re.IGNORECASE)
        result = pattern.sub(original, result)
    return result


# ---------- মূল অনুবাদ (ব্যাচ) ----------
def _translate_batch(
    texts: List[str],
    device: str = "cuda",
    batch_size: int = 16,
) -> List[str]:
    """IndicTrans2 দিয়ে একাধিক বাক্য একসাথে অনুবাদ।"""
    model, tokenizer, processor = _load_model(device)

    outputs: List[str] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        # ১) IndicProcessor দিয়ে প্রি-প্রসেস (entity extraction সহ)
        batch = processor.preprocess_batch(
            batch_texts, src_lang=SRC_LANG, tgt_lang=TGT_LANG
        )

        # ২) টোকেনাইজ
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        # ৩) জেনারেশন
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # ৪) ডিকোড
        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # ৫) পোস্ট-প্রসেস (entity restoration)
        translations = processor.postprocess_batch(decoded, lang=TGT_LANG)
        outputs.extend(translations)

    return outputs


# ---------- পাবলিক API (pipeline.py যেভাবে কল করে) ----------
def translate_sentences_google(
    sentences: List[Dict],
    glossary: List[str],
    device: str = "cuda",
    batch_size: int = 16,
) -> List[Dict]:
    """
    আগের google-translator ভার্সনের সাথে হুবহু একই আউটপুট ফরম্যাট।
    (ফাংশনের নাম ইচ্ছাকৃতভাবে একই রাখা — pipeline.py-তে বদল লাগবে না।)
    """
    if not sentences:
        return []

    # ১) Glossary placeholder প্রয়োগ
    prepared: List[Tuple[Dict, str, Dict[str, str]]] = []
    for item in sentences:
        masked, mapping = apply_glossary_placeholders(item["text"], glossary)
        prepared.append((item, masked, mapping))

    # ২) ব্যাচে অনুবাদ
    logger.info(f"🌐 IndicTrans2 দিয়ে {len(prepared)} বাক্য অনুবাদ করা হচ্ছে (batch={batch_size})...")
    results: List[Dict] = []

    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        src_texts = [b[1] for b in batch]

        try:
            translated_texts = _translate_batch(src_texts, device=device, batch_size=batch_size)
        except Exception as e:
            logger.error(f"❌ ব্যাচ অনুবাদ ব্যর্থ: {e} — মূল টেক্সট রাখা হচ্ছে।")
            translated_texts = src_texts  # fallback

        for idx, (orig_item, _, mapping) in enumerate(batch):
            raw = translated_texts[idx] if idx < len(translated_texts) else orig_item["text"]
            final_bn = restore_glossary_words(raw, mapping)
            results.append({
                "start": orig_item["start"],
                "end": orig_item["end"],
                "src_text": orig_item["text"],
                "tgt_text": final_bn,
            })

        done = min(start + batch_size, len(prepared))
        logger.info(f"   ✓ {done}/{len(prepared)} বাক্য অনূদিত")

    flush_memory()
    return results