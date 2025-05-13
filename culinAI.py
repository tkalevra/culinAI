import os
import sys
import time
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import argparse
import pyarrow as pa

# 🧀 Welcome to "DeepSeek CulinAI" - A Script So Refined, It Cooks Your VRAM
# Because why just train an AI when you can watch it sizzle?

# Ensure Python version is 3.10 or higher
assert sys.version_info >= (3, 10), "This script requires Python 3.10 or higher."

print(f"Using Python: {sys.executable}")

# Set base directory (relative to script location)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Text-Datasets'))
cache_dir = os.path.join(base_dir, 'HF_CACHE')
temp_dir = os.path.join(cache_dir, 'TEMP')

# Ensure cache and temp directories exist
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Set HuggingFace, Dataset, Arrow, and System Temporary directories
os.environ['HF_HOME'] = cache_dir
os.environ['DATASETS_CACHE'] = cache_dir
os.environ['ARROW_TEMP_DIR'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONHASHSEED'] = '0'
pa.set_memory_pool(pa.system_memory_pool())

# Display current environment paths
print(f"Base Directory: {base_dir}")
print(f"Cache Directory: {cache_dir}")
print(f"Temporary Directory: {temp_dir}")

# Disk space check
total, used, free = shutil.disk_usage(base_dir)
print(f"Free space on dataset drive: {free // (1024 ** 3)} GB")

# Clearing HuggingFace cache - Make some room for the spice!
print("Clearing HuggingFace cache...")

# Display VRAM Information - This is where the sizzle happens
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"Detected GPU: {gpu.name} with {gpu.total_memory / (1024 ** 3):.2f} GB VRAM")

# Load model - Welcome to the heat!
model_name = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

# Brief delay to ensure VRAM allocation
time.sleep(1)

# Display Accurate VRAM after model load
if torch.cuda.is_available():
    total_vram = gpu.total_memory / (1024 ** 3)
    real_allocated = min(torch.cuda.memory_allocated(), torch.cuda.memory_reserved()) / (1024 ** 3)
    free_vram = max(0, total_vram - real_allocated)
    print(f"VRAM - Real Allocated: {real_allocated:.2f} GB | Free: {free_vram:.2f} GB (of {total_vram:.2f} GB)")

# Dataset auto-detection - Scouting for ingredients
print("\nScanning for datasets...")
