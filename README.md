# Entropy-Aware Structural Alignment for Zero-Shot Handwritten Chinese Character Recognition

This repository contains the official implementation of the paper **"Entropy-Aware Structural Alignment for Zero-Shot Handwritten Chinese Character Recognition"**.



## 🎯 Introduction

Zero-Shot Handwritten Chinese Character Recognition is a challenging task due to the large number of Chinese characters and the scarcity of handwritten samples for rare characters. This work proposes an **Entropy-Aware Structural Alignment** method that:

- Leverages structural decomposition of Chinese characters
- Uses entropy-based weighting to focus on discriminative components
- Enables recognition of unseen characters through structural similarity


## 🔧 Dependencies

The project relies on the following key dependencies. All required packages are listed in `requirements.txt` for easy installation:

- Python 3.8 (recommended)

- PyTorch (1.10.0+)

### Environment Setup

Follow the steps below to set up the running environment:

1. Create a conda virtual environment (recommended to avoid dependency conflicts):
        conda create -n <your_env_name> python=3.8
        conda activate <your_env_name>

2. Install all required packages using pip:
        pip install -r requirements.txt

## Training

The training process is straightforward. Follow these steps to start training:

1. **Modify Configuration**: Edit the configuration file (located in the `config.py`) to set experimental parameters, such as dataset path, model hyperparameters, training epochs, learning rate, and experiment name (`exp_name`). 

2. **Start Training**: Use the following command to start distributed training. If you have a different number of GPUs, adjust the `--nproc_per_node` parameter accordingly:
        torchrun --nproc_per_node=4 train.py
Alternatively, you can use the provided shell script for one-click training:
        bash run.sh

## Results

All training results are automatically saved in the specified directory, which is determined by the `exp_name` parameter in the configuration file. The directory structure is as follows:

```text
./history/config['exp_name']
├── record.txt            # Final result of each epoch
├── train_log.txt         # Training process
├── result_file*.txt      # The character inference result
└── Other backups
```

## Project Structure
```text
./
├── train.py            # Main training script 
├── run.sh              
├── requirements.txt    # List of all required dependencies
├── config.py           # Configuration files 
├── history/            # Directory for saving all experimental results
├── model/              # Model definition directory
│   ├── ocr_encoder.py     # Encoder (ResNet)
│   └── ocr_decoder.py     # Decoder (Transformer-Decoder)
├── data/               # Character data related files
├── util.py              # Utility functions
├── radical_encoder/              # Pre-train CLIP radicals encoder
│   ├── clip_dataset.py     # Dataset class
│   ├── encoder_train.py    # Fine-tune CLIP
│   ├── tsne-kean.py        # Visualization
│   └── count_radical_entropy.py 
├── structure/              # Dual-View Structural Representation
│   ├── tree_structure_embedding.py     # Dual-View Structural Representation position information extraction
└── dataset/             # Put the lmdb dataset here
```

## Citation

If you use this code or the proposed method in your research, please cite the original paper. Replace the placeholder information with the actual paper details once published:

```text
@article{luo2026entropy,
  title={Entropy-Aware Structural Alignment for Zero-Shot Handwritten Chinese Character Recognition},
  author={Luo, Qiuming and Zeng, Tao and Li, Feng and Liu, Heming and Mao, Rui and Kong, Chang},
  journal={arXiv preprint arXiv:2602.03913},
  year={2026}
}
```