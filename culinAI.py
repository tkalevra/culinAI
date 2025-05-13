import os
import sys
import time
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import argparse
import pyarrow as pa
from tqdm import tqdm  # Progress bar

# Ensure Python version is 3.10 or higher
assert sys.version_info >= (3, 10), "This script requires Python 3.10 or higher."

print(f"Using Python: {sys.executable}")

# Set base directory (relative to script location)
base_dir = os.path.abspath(os.path.dirname(__file__))
cache_dir = os.path.join(base_dir, 'HF_CACHE')
temp_dir = os.path.join(base_dir, 'TEMP')

# Ensure all directories exist
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Set all cache, temp, and processing directories to local
os.environ['HF_HOME'] = cache_dir
os.environ['DATASETS_CACHE'] = cache_dir
os.environ['ARROW_TEMP_DIR'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir

def clear_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

def get_vram_info():
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)  # MB
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)  # MB
    free = total - reserved
    return total, allocated, free

def check_disk_space(required_gb=20):
    total, used, free = shutil.disk_usage(base_dir)
    free_gb = free / (1024 ** 3)  # Convert to GB
    print(f"Disk Space Available: {free_gb:.2f} GB")

    if free_gb < required_gb:
        print(f"❌ Warning: Less than {required_gb} GB of free disk space available.")
        choice = input(f"Continue anyway? (Y/N): ").strip().lower()
        if choice != 'y':
            print("❌ Operation aborted by user due to insufficient disk space.")
            sys.exit(1)
    else:
        print(f"✅ Sufficient disk space available.\n")

# Check Disk Space
check_disk_space(20)

# VRAM Control
vram_target = 0.9  # Start with 90% VRAM
model = None
model_name = "deepseek-ai/deepseek-llm-7b-base"

clear_vram()
total_vram, _, free_vram = get_vram_info()
print(f"VRAM Target: {vram_target * total_vram:.2f} MB | Available: {free_vram:.2f} MB")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Debug VRAM before model loading
clear_vram()
total_vram, used_vram, free_vram = get_vram_info()
print(f"Before Loading - VRAM: Total: {total_vram:.2f} MB | Used: {used_vram:.2f} MB | Free: {free_vram:.2f} MB")

try:
    # Load the model using accelerate for auto-offloading
    print("\n🔧 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    clear_vram()
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    sys.exit(1)

# Confirm VRAM usage after model load
total_vram, used_vram, free_vram = get_vram_info()
print(f"After Loading - VRAM: Total: {total_vram:.2f} MB | Used: {used_vram:.2f} MB | Free: {free_vram:.2f} MB")

if used_vram > vram_target * total_vram:
    print("❌ Model loaded, but VRAM exceeds target. Exiting.")
    sys.exit(1)

print("\n✅ Model loaded successfully.")

# Dataset Selection and Loading
print("\n🔍 Scanning for datasets...")
dataset_dir = os.path.join(base_dir, 'Text-Datasets')
available_datasets = {}

# Traverse directories for datasets
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(('.txt', '.json', '.csv', '.parquet')):
            parent_dir = os.path.basename(os.path.dirname(root))
            if parent_dir not in available_datasets:
                available_datasets[parent_dir] = []
            available_datasets[parent_dir].append(os.path.join(root, file))

if not available_datasets:
    print("❌ No datasets found. Exiting.")
    sys.exit(1)

# Display detected datasets
print("\n📊 Detected Datasets:")
for idx, dataset in enumerate(available_datasets.keys(), 1):
    print(f"{idx}. {dataset}")

# Argument parser for dataset selection
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="all", help="Specify a dataset name or use 'all'")
args = parser.parse_args()

selected_datasets = []

if args.dataset.lower() == "all":
    selected_datasets = list(available_datasets.keys())
else:
    if args.dataset in available_datasets:
        selected_datasets = [args.dataset]
    else:
        print(f"❌ Dataset '{args.dataset}' not found. Available options: {list(available_datasets.keys())}")
        sys.exit(1)

print(f"\n📦 Selected Datasets: {selected_datasets}\n")

# Load selected datasets with progress bar
loaded_datasets = []
total_samples = 0

for dataset_name in tqdm(selected_datasets, desc="Loading Datasets", unit="dataset"):
    print(f"\n🔍 Loading dataset: {dataset_name}")
    for file_path in tqdm(available_datasets[dataset_name], desc=f"{dataset_name}", unit="file"):
        try:
            print(f"Loading {file_path}...")
            dataset = load_dataset("parquet", data_files=file_path, cache_dir=cache_dir)
            train_samples = len(dataset['train']) if 'train' in dataset else len(dataset)
            print(f"✅ Loaded {os.path.basename(file_path)} with {train_samples} samples.")
            total_samples += train_samples
            loaded_datasets.append(dataset)
        except Exception as e:
            print(f"\n❌ Failed to load {os.path.basename(file_path)}: {str(e).splitlines()[-1]}")

if not loaded_datasets:
    print("❌ No datasets loaded. Exiting.")
    sys.exit(1)

print("\n✅ All Datasets Loaded Successfully")
print(f"📊 Total Loaded Samples: {total_samples:,}")
print(f"📦 Total Datasets Loaded: {len(loaded_datasets)}")
