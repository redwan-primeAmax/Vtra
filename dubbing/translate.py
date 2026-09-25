"""
NLLB-200-distilled-600M (INT8, CTranslate2) দিয়ে লোকাল অনুবাদ।
Google Translate-এর কোনো API/IP ব্যবহার হয় না — সম্পূর্ণ অফলাইন।
"""
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download, hf_hub_download
from dubbing.memory import flush_memory

logger = logging.getLogger("dubbing.translate")

# ---------- NLLB-200 ভাষা কোড ----------
SRC_LANG = "eng_Latn"
TGT_LANG = "ben_Beng"

# ---------- মডেল রেপো ----------
# INT8 pre-converted CT2 মডেল (RAM ~1.2GB)
CT2_MODEL_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
# টোকেনাইজার এই রেপো থেকে নেওয়া হবে
TOKENIZER_REPO = "facebook/nllb-200-distilled-600M"

# ---------- সিঙ্গেলটন (মডেল শুধু একবার লোড হবে) ----------
_translator: Optional[ctranslate2.Translator] = None
_tokenizer: Optional[spm.SentencePieceProcessor] = None


def _resolve_device(device: str) -> Tuple[str, str]:
    """device স্ট্রিং → (ct2_device, compute_type)"""
    if device in ("cuda", "gpu"):
        return "cuda", "int8_float16"   # GPU-তে INT8 + FP16
    return "cpu", "int8"                # CPU-তে pure INT8


def _load_model(device: str = "cpu"):
    """একবারই মডেল লোড হবে (singleton)। দ্বিতীয়বার কল করলে ক্যাশ থেকে দেবে।"""
    global _translator, _tokenizer
    if _translator is not None and _tokenizer is not None:
        return _translator, _tokenizer

    ct2_device, compute_type = _resolve_device(device)
    logger.info(f"📥 NLLB-200 INT8 লোড হচ্ছে (device={ct2_device}, compute={compute_type})...")

    # ১) CT2 মডেল ডাউনলোড (ক্যাশ হবে ~/.cache/huggingface)
    model_dir = snapshot_download(
        repo_id=CT2_MODEL_REPO,
        allow_patterns=["*.bin", "*.json", "*.txt", "*.model"],
    )

    # ২) Translator তৈরি
    _translator = ctranslate2.Translator(
        model_dir,
        device=ct2_device,
        compute_type=compute_type,
        inter_threads=2,
        intra_threads=4,
    )

    # ৩) SentencePiece টোকেনাইজার (মূল facebook রেপো থেকে)
    spm_path = hf_hub_download(TOKENIZER_REPO, "sentencepiece.bpe.model")
    _tokenizer = spm.SentencePieceProcessor()
    _tokenizer.load(spm_path)

    logger.info("✅ NLLB-200 মডেল প্রস্তুত")
    return _translator, _tokenizer


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
        # অনুবাদক কখনো placeholder-এর ভেতরে স্পেস বসায়, তাই loosened regex
        pattern = re.compile(r"\s*".join(re.escape(c) for c in placeholder), re.IGNORECASE)
        result = pattern.sub(original, result)
    return result


# ---------- মূল অনুবাদ (ব্যাচ) ----------
def _translate_batch(
    texts: List[str],
    device: str = "cpu",
    beam_size: int = 2,
    max_batch_size: int = 16,
) -> List[str]:
    """একাধিক বাক্য একসাথে অনুবাদ করে — প্রতি বাক্যে আলাদা ইনফারেন্সের চেয়ে অনেক দ্রুত।"""
    translator, sp = _load_model(device)

    # টোকেনাইজ: প্রতিটি বাক্যের আগে সোর্স ভাষা ট্যাগ
    tokenized: List[List[str]] = []
    for text in texts:
        pieces = sp.encode(text, out_type=str)
        tokenized.append([SRC_LANG] + pieces + ["</s>"])

    # ব্যাচ অনুবাদ, প্রতিটির টার্গেট প্রিফিক্স বাংলা
    batch_results = translator.translate_batch(
        tokenized,
        target_prefix=[[TGT_LANG]] * len(tokenized),
        beam_size=beam_size,
        max_batch_size=max_batch_size,
        num_hypotheses=1,
        return_scores=False,
    )

    outputs: List[str] = []
    for res in batch_results:
        hyp = res.hypotheses[0]
        # প্রথম টোকেন হলো ভাষা ট্যাগ — সরিয়ে দিই
        if hyp and hyp[0] == TGT_LANG:
            hyp = hyp[1:]
        # </s> থাকলে বাদ
        if "</s>" in hyp:
            hyp = hyp[: hyp.index("</s>")]
        text_out = sp.decode(hyp).strip()
        outputs.append(text_out)

    return outputs


# ---------- পাবলিক API (pipeline.py যেভাবে কল করে) ----------
def translate_sentences_google(
    sentences: List[Dict],
    glossary: List[str],
    device: str = "cpu",
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
    logger.info(f"🌐 NLLB-200 দিয়ে {len(prepared)} বাক্য অনুবাদ করা হচ্ছে (batch={batch_size})...")
    results: List[Dict] = []

    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        src_texts = [b[1] for b in batch]

        try:
            translated_texts = _translate_batch(src_texts, device=device)
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