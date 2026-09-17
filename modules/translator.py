import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config.settings import Config
from utils.logger import logger

class BengaliTranslator:
    def __init__(self):
        self.device = getattr(Config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        model_name = getattr(Config, "TRANSLATOR_MODEL", "facebook/nllb-200-distilled-600M")
        logger.info("NLLB-200 অনুবাদ মডিউল জিপিইউতে লোড হচ্ছে...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def translate_batch(self, text_list):
        if not text_list:
            return []
        
        clean_texts = []
        for item in text_list:
            if isinstance(item, dict):
                clean_texts.append(str(item.get('text', '')))
            elif hasattr(item, 'text'):
                clean_texts.append(str(item.text))
            else:
                clean_texts.append(str(item))

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids("ben_Beng")
        inputs = self.tokenizer(clean_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=256
            )
        
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

_translator_instance = None

def translate_en_to_bn(text_data):
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = BengaliTranslator()
    
    if isinstance(text_data, str):
        res = _translator_instance.translate_batch([text_data])
        return res[0] if res else text_data
    
    elif isinstance(text_data, list):
        # Extract texts only for translation
        texts = []
        for seg in text_data:
            if isinstance(seg, dict):
                texts.append(seg.get('text', ''))
            elif hasattr(seg, 'text'):
                texts.append(seg.text)
            else:
                texts.append(str(seg))
        
        translated_texts = _translator_instance.translate_batch(texts)
        
        # ⭐ Merge translated text back with original start/end timestamps
        # This preserves ALL timing information for downstream sync operations
        merged = []
        for seg, tr_text in zip(text_data, translated_texts):
            if isinstance(seg, dict):
                merged.append({
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": seg.get("text", ""),
                    "translated_text": tr_text
                })
            elif hasattr(seg, 'start') and hasattr(seg, 'end'):
                merged.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": getattr(seg, 'text', ''),
                    "translated_text": tr_text
                })
            else:
                merged.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": str(seg),
                    "translated_text": tr_text
                })
        return merged
    
    else:
        res = _translator_instance.translate_batch([str(text_data)])
        return res[0] if res else str(text_data)