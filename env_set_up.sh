#!/bin/bash

# Create a new conda environment named allin
conda create --name allin python=3.12 -y

# Activate the conda environment
source activate allin

# Install the required libraries using conda
conda install -y numpy pandas scikit-learn matplotlib requests seaborn tqdm

# Install the remaining libraries using pip
pip install mutagen torch torchvision torchaudio imbalanced-learn