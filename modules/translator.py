import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import Config
from utils.logger import logger
from utils.gpu import clear_vram

class TranslatorEngine:
    def __init__(self, model_id: str = Config.TRANSLATION_MODEL_NAME):
        """
        Qwen2.5-3B-Instruct মডেল লোড করে যা ইংরেজি থেকে সাবলীল বাংলা অনুবাদের জন্য উপযোগী।
        """
        try:
            logger.info(f"Qwen অনুবাদ মডেল লোড হচ্ছে: {model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            logger.info("Qwen Translator Model সফলভাবে লোড হয়েছে।")
        except Exception as e:
            logger.error(f"Qwen Model লোড করতে ব্যর্থ: {e}")
            raise e

    def translate_text(self, text: str) -> str:
        """
        Qwen LLM ব্যবহার করে ইংরেজি টেক্সটকে প্রাকৃতিকভাবে বাংলায় অনুবাদ করে।
        """
        if not text or not text.strip():
            return ""

        try:
            prompt = (
                "Translate the following English text to natural, spoken Bengali. "
                "Only return the translated Bengali text, nothing else.\n\n"
                f"English: {text}\nBengali:"
            )

            messages = [
                {"role": "system", "content": "You are a professional English to Bengali translator for video dubbing."},
                {"role": "user", "content": prompt}
            ]

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            translated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return translated_text.strip()

        except Exception as e:
            logger.error(f"অনুবাদ করতে গিয়ে ত্রুটি ঘটেছে: {e}")
            return text

# main.py এর জন্য হেলপার ফাংশনসমূহ
_translator_instance = None

def get_translator_engine() -> TranslatorEngine:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = TranslatorEngine()
    return _translator_instance

def translate_en_to_bn(segments: list[dict]) -> list[dict]:
    """
    main.py থেকে সরাসরি কল করার জন্য ফাংশন ইন্টারফেস।
    """
    translator = get_translator_engine()
    translated_segments = []

    for seg in segments:
        original_text = seg.get("text", "")
        translated_text = translator.translate_text(original_text)
        
        new_seg = dict(seg)
        new_seg["translated_text"] = translated_text
        translated_segments.append(new_seg)

    clear_vram()
    return translated_segments

def save_translation_to_txt(translated_segments: list[dict], output_path: str):
    """
    অনুবাদ এনালাইসিস ফাইলে সেভ করার হেলপার ফাংশন।
    """
    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("      TRANSLATION ANALYSIS (ENGLISH -> BENGALI)   \n")
            f.write("==================================================\n\n")
            
            for idx, seg in enumerate(translated_segments, 1):
                start = seg.get('start', 0.0)
                end = seg.get('end', 0.0)
                original_text = seg.get('text', '')
                translated_text = seg.get('translated_text', '')
                
                f.write(f"[{idx:03d}] TIME: {start:.2f}s -> {end:.2f}s\n")
                f.write(f"EN : {original_text}\n")
                f.write(f"BN : {translated_text}\n")
                f.write("-" * 50 + "\n")
                
        logger.info(f"অনুবাদ এনালাইসিস ফাইল সফলভাবে সেভ হয়েছে: {output_path}")
    except Exception as e:
        logger.error(f"অনুবাদ এনালাইসিস ফাইল সেভ করতে সমস্যা হয়েছে: {e}")
