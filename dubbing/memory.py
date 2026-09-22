import gc
import torch
import psutil
import logging

logger = logging.getLogger("dubbing.memory")

def flush_memory():
    """VRAM এবং System RAM সম্পূর্ণ খালি করে।"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    ram_gb = psutil.virtual_memory().used / (1024 ** 3)
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        logger.info(f"[Memory Cleared] System RAM: {ram_gb:.2f} GB | VRAM Allocated: {vram_allocated:.2f} GB")
