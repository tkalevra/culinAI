import os
import sys
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import argparse
import pyarrow as pa
from tqdm import tqdm

# Ensure Python version is 3.10 or higher
assert sys.version_info >= (3, 10), "This script requires Python 3.10 or higher."

print(f"Using Python: {sys.executable}")

# Set base directory (relative to script location)
base_dir = os.path.abspath(os.path.dirname(__file__))
cache_dir = os.path.join(base_dir, 'HF_CACHE')
temp_dir = os.path.join(base_dir, 'TEMP')
offload_dir = os.path.join(cache_dir, 'offload')

# Ensure all directories exist
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(offload_dir, exist_ok=True)

# Set all cache, temp, and processing directories to local
os.environ['HF_HOME'] = cache_dir
os.environ['DATASETS_CACHE'] = cache_dir
os.environ['ARROW_TEMP_DIR'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir

# Minimum disk space threshold in GB (20 GB default)
MIN_DISK_SPACE_GB = 20

# Function to check disk space
def check_disk_space(path, min_gb=MIN_DISK_SPACE_GB):
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    if free_gb < min_gb:
        print(f"❗ Warning: Low disk space on {path}. Only {free_gb:.2f} GB free. Minimum required: {min_gb} GB.")
        choice = input("Continue anyway? (Y/N): ").strip().lower()
        if choice != 'y':
            print("❌ Exiting due to insufficient disk space.")
            sys.exit(1)
    else:
        print(f"✅ Disk space check passed: {free_gb:.2f} GB free.")

# Check disk space before proceeding
check_disk_space(base_dir, MIN_DISK_SPACE_GB)

# Function to clear VRAM
def clear_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

# Function to get VRAM details
def get_vram_info():
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)  # MB
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)  # MB
    free = total - reserved
    return total, allocated, free

# Scan for datasets
print("\n🔍 Scanning for datasets...")
dataset_dir = os.path.join(base_dir, 'Text-Datasets')
all_datasets = []

for root, dirs, files in os.walk(dataset_dir):
    if any(file.endswith(('.txt', '.json', '.csv', '.parquet')) for file in files):
        all_datasets.append(os.path.basename(root))

if not all_datasets:
    print("❌ No datasets detected. Please ensure your datasets are in the 'Text-Datasets' folder.")
    sys.exit(1)

print("\n📊 Detected the following datasets:")
for idx, dataset in enumerate(all_datasets, 1):
    print(f"{idx}. {dataset}")

selected_dataset = input("\nEnter the number of the dataset you want to use (or type 'all' for all): ").strip().lower()

if selected_dataset == 'all':
    selected_datasets = all_datasets
else:
    try:
        selected_datasets = [all_datasets[int(selected_dataset) - 1]]
    except (ValueError, IndexError):
        print("❌ Invalid selection. Exiting.")
        sys.exit(1)

print(f"\n✅ Selected Datasets: {', '.join(selected_datasets)}")

# Smart VRAM Control
vram_target = 0.9  # Start with 90% VRAM
model = None
model_name = "deepseek-ai/deepseek-llm-7b-base"
attempts = 0
max_attempts = 10
use_offload = False
use_safetensors = True  # Prioritize SafeTensors if available

clear_vram()
total_vram, used_vram, free_vram = get_vram_info()
target_vram = int(total_vram * vram_target)

print(f"\nTotal VRAM: {total_vram:.2f} MB | Free: {free_vram:.2f} MB")
print(f"VRAM Target for Model: {target_vram:.2f} MB")

while attempts < max_attempts:
    clear_vram()
    print(f"\nAttempting to load model with {vram_target * 100:.0f}% VRAM... (Attempt {attempts + 1})")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=use_safetensors,
            offload_folder=offload_dir if use_offload else None
        )

        clear_vram()
        total_vram, used_vram, free_vram = get_vram_info()
        print(f"VRAM - Allocated: {used_vram:.2f} MB | Free: {free_vram:.2f} MB (of {total_vram:.2f} MB)")

        if used_vram <= target_vram:
            print(f"✅ Model loaded using {target_vram} MB VRAM.")
            break
        else:
            print(f"❌ VRAM exceeds target. Adjusting...")
            vram_target -= 0.05
            target_vram = int(total_vram * vram_target)
            attempts += 1
            model = None

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("❌ Out of memory. Adjusting VRAM target.")
            vram_target -= 0.05
            target_vram = int(total_vram * vram_target)
            attempts += 1
            model = None
        elif 'safetensors' in str(e).lower():
            print("⚠️ SafeTensors failed. Disabling SafeTensors.")
            use_safetensors = False
        else:
            raise e

if not model:
    print("❌ Unable to load model after maximum retries. Exiting.")
    sys.exit(1)
