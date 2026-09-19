import gc
import torch
import psutil
import logging

logger = logging.getLogger("dubbing.memory")

def flush_memory():
    """VRAM এবং RAM সম্পূর্ণ পরিষ্কার করে।"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def log_memory_usage():
    ram_gb = psutil.virtual_memory().used / (1024 ** 3)
    logger.info(f"System RAM: {ram_gb:.2f} GB")
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"VRAM Allocated: {vram_allocated:.2f} GB | Reserved: {vram_reserved:.2f} GB")
