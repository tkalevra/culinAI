import os
import sys
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from datasets import load_dataset
import argparse
import pyarrow as pa
from tqdm import tqdm
import json

# Ensure Python version is 3.10 or higher
assert sys.version_info >= (3, 10), "This script requires Python 3.10 or higher."

print(f"Using Python: {sys.executable}")

# Hardware Detection
if torch.cuda.is_available():
    device = torch.cuda.get_device_properties(0)
    print(f"✅ Detected GPU: {device.name}")
    print(f"✅ Total VRAM: {device.total_memory / (1024 ** 3):.2f} GB")
    vram_target = device.total_memory * 0.9 / (1024 ** 3)
    print(f"✅ VRAM Target: {vram_target:.2f} GB")
else:
    print("❌ No CUDA device detected. Using CPU.")

# Set base directory (relative to script location)
base_dir = os.path.abspath(os.path.dirname(__file__))
cache_dir = os.path.join(base_dir, 'HF_CACHE')
temp_dir = os.path.join(base_dir, 'TEMP')
offload_dir = os.path.join(cache_dir, 'offload')

# Ensure all directories exist
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(offload_dir, exist_ok=True)

# Model selection
model_base_dir = "D:\\DeepSeek-V3"
MODEL_PATHS = {}

for folder in os.listdir(model_base_dir):
    model_path = os.path.join(model_base_dir, folder)
    if os.path.isdir(model_path):
        MODEL_PATHS[folder] = model_path

# Model selection
print("Available models:")
for idx, model in enumerate(MODEL_PATHS.keys(), 1):
    print(f"{idx}. {model}")

if not MODEL_PATHS:
    print("❌ No models detected in the specified directory. Exiting.")
    sys.exit(1)

model_choice = input("Select a model by number: ").strip()
try:
    selected_model = list(MODEL_PATHS.values())[int(model_choice) - 1]
except (ValueError, IndexError):
    print("Invalid selection. Exiting.")
    sys.exit(1)

print(f"Selected model: {selected_model}")

# Config correction with backup
config_path = os.path.join(selected_model, 'config.json')
if os.path.exists(config_path):
    config_backup = config_path + ".BAK"
    shutil.copy2(config_path, config_backup)
    print(f"✅ Original config.json backed up as {config_backup}")

    with open(config_path, 'r') as file:
        config_data = json.load(file)

    if "rope_scaling" in config_data:
        scaling = config_data.get("rope_scaling", {})
        for key in list(scaling.keys()):
            if key in ["factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim", "original_max_position_embeddings"]:
                try:
                    scaling[key] = float(scaling[key])
                except (ValueError, TypeError):
                    print(f"⚠️ Invalid value for {key} in rope_scaling. Setting default float 1.0.")
                    scaling[key] = 1.0
            elif key == "type":
                scaling[key] = str(scaling[key])  # Ensure type is always a string

        config_data["rope_scaling"] = scaling
        print("✅ Rope scaling values corrected to: ", scaling)

        with open(config_path, 'w') as file:
            json.dump(config_data, file, indent=4)
            print("✅ Corrected config.json saved to disk.")

# Load Model
clear_vram = lambda: (torch.cuda.empty_cache(), torch.cuda.ipc_collect()) if torch.cuda.is_available() else None
clear_vram()

print(f"🚀 Loading model from: {selected_model}")
model = AutoModelForCausalLM.from_pretrained(
    selected_model,
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(selected_model, cache_dir=cache_dir)

print("✅ Model and tokenizer loaded successfully.")
