import torch
import psutil
import gc

def get_gpu_memory_usage():
    """Get GPU memory usage in GB"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        return memory_allocated, memory_reserved
    return 0, 0

def calculate_batch_size(gpu_memory, min_size=1):
    """Calculate optimal batch size based on available GPU memory"""
    reserved_memory = 4
    usable_memory = max(0, gpu_memory - reserved_memory)
    memory_per_sample = 0.5
    max_batch = max(min_size, min(16, int(usable_memory / memory_per_sample)))
    return max_batch

def log_memory_usage():
    """Log both CPU and GPU memory usage"""
    cpu_percent = psutil.Process().memory_percent()
    gpu_allocated, gpu_reserved = get_gpu_memory_usage()
    print(f"Memory Usage - CPU: {cpu_percent:.1f}%, GPU Allocated: {gpu_allocated:.1f}GB, Reserved: {gpu_reserved:.1f}GB")

def clean_memory():
    """Clean up GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
