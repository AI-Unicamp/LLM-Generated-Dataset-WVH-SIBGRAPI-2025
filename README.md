# LLM-Generated Dataset for Speech-Driven 3sD Facial Animation Models with Text-Controlled Expressivity

This repository contains the implementation and datasets for generating synthetic facial animation data using Large Language Models (LLMs) with text-controlled expressivity for 3D facial animation models.

## 📁 Repository Structure

```
├── dataframes/          # Processed emotion datasets
├── gen_data/            # Generated synthetic datasets
├── raw_data/            # Original emotion datasets
├── scripts/             # Main implementation scripts
│   ├── clip_module/     # CLIP-based model training
│   ├── dataset_generation/  # LLM-based data generation
│   └── evaluation/      # Model evaluation and visualization
├── environment.yml      # Conda environment configuration
└── requirements.txt     # Python dependencies
```

## 🎯 Project Overview

This project focuses on creating high-quality synthetic datasets for training speech-driven 3D facial animation models. The approach combines:

- **Multi-source emotion datasets** (GoEmotions, Tweet Intensity, ISEAR)
- **LLM-generated facial descriptions** using Llama 3.3 70B
- **CLIP-based multimodal alignment** between text and facial blendshapes
- **Action Unit (AU) mapping** based on FACS (Facial Action Coding System)

## 🚀 Quick Start

### Prerequisites

- CUDA-compatible GPU (recommended)
- Conda or Python 3.9+
- Git LFS for model weights

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/AI-Unicamp/LLM-Generated-Dataset.git
cd LLM-Generated-Dataset
```

2. **Install Git LFS (required for model weights):**
```bash
sudo apt install git-lfs
conda install git-lfs
git lfs pull
```

3. **Set up environment:**
```bash
# Using Conda (recommended)
conda env create -f environment.yml
conda activate llm_generated_dataset

# Or using pip
pip install -r requirements.txt
```

## 📊 Datasets

### Input Datasets (raw_data/)
- **GoEmotions**: 58k Reddit comments with emotion labels
- **Tweet Intensity**: Emotion intensity tweets (anger, fear, joy, sadness)
- **ISEAR**: International Survey on Emotion Antecedents and Reactions

### Generated Datasets (gen_data/)
- **Final synthetic dataset**: Text + emotions + descriptions + blendshapes
- **LLM outputs**: Llama 3.3 70B generated emotional descriptions and action units

## 🧠 Model Architecture

### CLIP Module (`scripts/clip_module/`)

The core training pipeline includes:

- **BlendshapeEncoder**: Encodes 51D blendshape vectors to latent space
- **TextProjector**: Projects CLIP text embeddings to shared latent space  
- **BlendshapeDecoder**: Reconstructs blendshapes from latent representations
- **ClipEncoderModule**: Frozen CLIP model for text encoding

### Key Components:

```python
# Model initialization
encoder = BlendshapeEncoder()
decoder = BlendshapeDecoder() 
projector = TextProjector()
clip_encoder = ClipEncoderModule()

# Training with multimodal alignment
trainer = Trainer(
    encoder=encoder,
    decoder=decoder, 
    projector=projector,
    clip_encoder=clip_encoder,
    dataset=dataset,
    batch_size=256,
    learning_rate=1e-5,
    epochs=100
)
```

## 🔧 Usage

### 1. Dataset Generation

Generate emotion datasets from raw sources:
```bash
cd scripts/dataset_generation/
python gen_dataframe_goemo.py
python gen_dataframe_tweet.py
python gen_dataframe_isear.py
python gen_dataframe_final.py
```

### 2. LLM-based Augmentation

Generate facial descriptions using Llama 3.3:
```bash
# Configure your HuggingFace token in get_token.py
python gen_dataset_llama33_4bit.py
```

### 3. Model Training

Train the CLIP-based alignment model:
```bash
cd scripts/clip_module/
python main.py
```

### 4. Evaluation

Generate t-SNE visualizations:
```bash
cd scripts/evaluation/
python tsne_plot.py
```

## 📄 Citation

If you use this code or dataset in your research, please cite:

```bibtex
TBD
```


## 📞 Contact

For questions or collaboration opportunities, please reach out through:
- GitHub Issues
- Email: p243236@dac.unicamp.br
- Institution: AIMS-Unicamp
