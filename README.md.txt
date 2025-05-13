# DeepSeek CulinAI

Welcome to **DeepSeek CulinAI**, a high-performance, customizable machine learning training script, designed to train language models using Hugging Face Transformers.

## About
DeepSeek CulinAI is a robust Python script designed for training language models using the Hugging Face Transformers library. It offers automatic dataset detection, caching, and efficient model loading.

## Features
- Automatic dataset scanning and loading (TXT, JSON, CSV, Parquet)
- Supports multi-dataset training (Wikitext, OpenWebText, The Pile, and more)
- Smart caching and temp directory management to avoid disk overflow
- Clear, detailed console output
- Customizable batch size and gradient accumulation

## Requirements
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
```

## Installation
1. Ensure Python 3.10+ is installed on your system.
2. Install CUDA (if using an NVIDIA GPU) and verify it's accessible by PyTorch.
3. Clone this repository and navigate to the directory.
4. Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository.
2. Place your text datasets in the `Text-Datasets` directory.
3. Ensure you have enough free disk space in the dataset directory.
4. Run the script with the desired dataset(s):

```bash
python train_lora.py --dataset all
```

## Command-Line Options
- `--dataset all` - Uses all detected datasets.
- `--dataset <name>` - Uses a specific dataset by name.

## Directory Structure
```
Text-Datasets/
├── HF_CACHE/ (automatically managed)
└── [Your Datasets]
```

## Advanced Tips
- Use the `requirements.txt` to maintain package versions.
- Adjust batch size and gradient accumulation directly in the script if you encounter OOM errors.
- CUDA setup must match the installed PyTorch version.

## License
This project is licensed under the MIT License. See the script header for full details.

## Credits
Created by The Spicy Coder Collective.
