import logging
import warnings
from main import process_videos

# ২. ক্লিন লগিং কনফিগারেশন (অপ্রয়োজনীয় ওয়ার্নিং ফিল্টার করা)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("dubbing.memory").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# শুধুমাত্র পরিষ্কার মেসেজ আউটপুট ফরম্যাট
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

if __name__ == "__main__":
    process_videos()
