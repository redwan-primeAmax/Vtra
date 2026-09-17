from deepmultilingualpunctuation import PunctuationModel
from utils.logger import setup_logger

logger = setup_logger()

class PunctuationEngine:
    def __init__(self):
        """
        হালকা Multilingual Punctuation Model লোড করে।
        এটি ট্রান্সক্রিপ্টের টেক্সটে কমা, ফুলস্টপ ও প্রশ্নবোধক চিহ্ন বসাবে।
        """
        try:
            logger.info("Punctuation Restoration Model লোড হচ্ছে (oliverguhr/fullstop-punctuation-multilingual-sonar)...")
            self.model = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilingual-sonar")
            logger.info("Punctuation Model সফলভাবে লোড হয়েছে।")
        except Exception as e:
            logger.error(f"Punctuation Model লোড করতে ব্যর্থ: {e}")
            raise e

    def restore_punctuation(self, text: str) -> str:
        """
        ইনপুট টেক্সটে বিরামচিহ্ন বসিয়ে রিটার্ন করে।
        """
        if not text or not text.strip():
            return text

        try:
            logger.info("টেক্সটে বিরামচিহ্ন (Punctuation) যোগ করা হচ্ছে...")
            clean_text = self.model.restore_punctuation(text)
            return clean_text
        except Exception as e:
            logger.error(f"Punctuation প্রসেস করতে সমস্যা হয়েছে: {e}")
            return text
