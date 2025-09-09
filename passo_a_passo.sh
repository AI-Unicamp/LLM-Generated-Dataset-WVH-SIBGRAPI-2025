sudo apt install git-lfs
conda install git-lfs

git clone https://github.com/AI-Unicamp/SLM-ER-Evaluation.git
cd LLM-Generated-Dataset-WVH-SIBGRAPI-2025

conda env create environment.yml
cd scripts
python -m evaluation.tsne_plot.py