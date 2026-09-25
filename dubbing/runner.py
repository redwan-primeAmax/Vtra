"""
শুধু environment variable ও logging setup।
main.py এটাকে import করবে (কোনো process_videos কল এখানে নেই)।
"""
import os
import logging
import warnings

# ---------- Environment cleanup ----------
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

# ---------- অপ্রয়োজনীয় ওয়ার্নিং ফিল্টার ----------
for _name in [
    "httpx", "huggingface_hub", "faster_whisper", "dubbing.memory",
    "transformers", "torch", "urllib3", "filelock", "tensorflow",
    "dubbing.translate", "dubbing.predownload",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

# ---------- পরিষ্কার আউটপুট ফরম্যাট ----------
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)