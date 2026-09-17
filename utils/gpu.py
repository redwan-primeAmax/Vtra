import gc
import torch

def clear_vram():
    """VRAM খালি করার মেমোরি ক্লিয়ারেন্স ফাংশন"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
