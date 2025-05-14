# DeepSeek CulinAI

Welcome to **DeepSeek CulinAI**, a high-performance, customizable machine learning training script, designed to train language models using Hugging Face Transformers.

## 🚀 About  
DeepSeek CulinAI is a robust, adaptive Python script designed for training language models using the Hugging Face Transformers library. It automatically detects datasets, dynamically manages VRAM, and optimizes model loading for smooth and efficient training.

## 🌟 Features  
- ✅ **Automatic Dataset Detection:** Scans and lists available datasets (TXT, JSON, CSV, Parquet).  
- ✅ **Multi-Dataset Training:** Supports training on Wikitext, OpenWebText, The Pile, and other datasets.  
- ✅ **Dynamic VRAM Management:** Automatically adjusts VRAM allocation to prevent out-of-memory (OOM) errors.  
- ✅ **Safe Disk Usage:** Verifies available disk space (default 20 GB) and warns if insufficient.  
- ✅ **Efficient Model Loading:** Supports SafeTensors for optimized model loading.  
- ✅ **Clear Console Output:** Improved clarity in progress and error messages.  

## 📌 Requirements  
- Python 3.10+  
- CUDA-enabled GPU (Tested on NVIDIA 4070 SUPER)  
- Required Python Packages:  

```plaintext
# requirements.txt  
torch>=2.0  
transformers>=4.30  
datasets>=2.13  
pyarrow>=10.0  
argparse>=1.4  
tqdm>=4.64  
```

## 📥 Installation  
1. Ensure Python 3.10+ is installed on your system.  
2. Install CUDA (if using an NVIDIA GPU) and verify it's accessible by PyTorch.  
3. Clone this repository and navigate to the directory.  
4. Install the required Python packages using:  

```bash
pip install -r requirements.txt
```

## 🚦 Usage  
1. Clone the repository.  
2. Place your text datasets in the `Text-Datasets` directory.  
3. Ensure you have enough free disk space in the dataset directory.  
4. Run the script with the desired dataset(s):

```bash
python train_lora.py --dataset all
```

## ⚙️ Command-Line Options  
- `--dataset all` - Uses all detected datasets.  
- `--dataset <name>` - Uses a specific dataset by name.  
- `--disk-threshold <size>` - Set minimum disk space (GB) required before training starts (default: 20 GB).  

## 📁 Directory Structure  
```
Text-Datasets/  
├── HF_CACHE/ (automatically managed)  
├── TEMP/ (temporary files, auto-cleaned)  
└── [Your Datasets]  
```

## 🚀 Advanced Tips  
- Use the `requirements.txt` to maintain package versions.  
- Adjust batch size and gradient accumulation directly in the script if you encounter OOM errors.  
- Use SafeTensors models for optimized memory usage.  
- CUDA setup must match the installed PyTorch version.  

## 📜 License  
This project is licensed under the MIT License. See the LICENSE file for full details.

## ❤️ Credits  
Created by The Spicy Coder Collective. 🚀  
