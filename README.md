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
├── requirements.txt     # Python dependencies
└── passo_a_passo.sh    # Setup and execution script
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

## 📈 Features

### Text Augmentation
- **Synonym replacement** using WordNet
- **Random word swapping** and deletion
- **Contextual perturbations** for robustness

### Blendshape Processing
- **51D facial blendshape vectors** (excluding head pose)
- **Noise injection** for data augmentation
- **FACS-based action unit mapping**

### Multimodal Training
- **Reconstruction loss**: MSE between original and reconstructed blendshapes
- **Alignment loss**: Cosine similarity between text and blendshape embeddings
- **Cross-modal loss**: Text-to-blendshape generation quality

## 🎨 Visualization

The evaluation module provides:
- **t-SNE embeddings** of text and blendshape representations
- **Arrow path visualization** showing semantic interpolation
- **Multi-modal embedding comparison**

Example output:
```python
# Generate t-SNE visualization
embeddings_dict = generator.get_embeddings()
tsne_results = generator.apply_tsne_to_all(embeddings_dict)
generator.plot_tsne_comparison(tsne_results)
```

## 📋 Configuration

### Model Hyperparameters
- **Latent dimension**: 512
- **Batch size**: 256
- **Learning rate**: 1e-5
- **Training epochs**: 100-200
- **Loss weights**: Reconstruction (1.0), Alignment (10.0), Cross-modal (10.0)

### Text Augmentation Settings
- **Synonym probability**: 0.1
- **Swap probability**: 0.1
- **Augmentations per sample**: 4

## 🤖 Pre-trained Models

Pre-trained model weights are available via Git LFS:
- `bs_encoder_weights_200ep_augmented.pth` (103MB)
- `txt_proj_weights_200ep_augmented.pth` (4MB)
- `bs_decoder_weights_200ep_augmented.pth` (103MB)

## 🔬 Research Applications

This work enables:
- **Controllable facial animation** from text descriptions
- **Emotion-aware speech synthesis** 
- **Cross-modal facial expression transfer**
- **Synthetic training data generation** for animation models

## 📄 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{llm_facial_animation_2025,
  title={LLM-Generated Dataset for Speech-Driven 3D Facial Animation Models with Text-Controlled Expressivity},
  author={Your Name and Collaborators},
  booktitle={SIBGRAPI 2025},
  year={2025}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AIMS-Unicamp** for research support
- **Meta AI** for Llama 3.3 model access
- **OpenAI** for CLIP model architecture
- **Emotion dataset creators**: GoEmotions, Tweet Intensity, ISEAR teams

## 📞 Contact

For questions or collaboration opportunities, please reach out through:
- GitHub Issues
- Email: [Your Email]
- Institution: AIMS-Unicamp

---

**Note**: This project requires significant computational resources for LLM inference and model training. Consider using cloud computing services for large-scale experiments.
