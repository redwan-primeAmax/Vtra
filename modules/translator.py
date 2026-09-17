import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.logger import setup_logger

logger = setup_logger()

class TranslatorEngine:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        """
        Qwen2.5-3B-Instruct মডেল লোড করে যা ইংরেজি থেকে সাবলীল বাংলা অনুবাদের জন্য উপযোগী।
        """
        try:
            logger.info(f"Qwen অনুবাদ মডেল লোড হচ্ছে: {model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # GPU থাকলে float16 ব্যবহার করবে, নতুবা float32
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
        Qwen LLM ব্যবহার করে ইংরেজি টেক্সটকে প্রাকৃতিকে বাংলায় অনুবাদ করে।
        """
        if not text or not text.strip():
            return ""

        try:
            # প্রম্পট তৈরি
            prompt = (
                "Translate the following English text to natural, spoken Bengali. "
                "Only return the translated Bengali text, nothing else.\n\n"
                f"English: {text}\nBengali:"
            )

            messages = [
                {"role": "system", "content": "You are a professional English to Bengali translator for video dubbing."},
                {"role": "user", "content": prompt}
            ]

            # চ্যাট টেমপ্লেট ফরম্যাট করা
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # টোকেনাইজ করা
            inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.model.device)

            # অনুবাদ জেনারেট করা
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            # আউটপুট ডিকোড করা
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            translated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return translated_text.strip()

        except Exception as e:
            logger.error(f"অনুবাদ করতে গিয়ে ত্রুটি ঘটেছে: {e}")
            return text
