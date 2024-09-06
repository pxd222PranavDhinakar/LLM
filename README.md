# PyTorch Implementations from Language Models Class

This repository contains various PyTorch implementations of machine learning models and techniques explored in my Language Models class. The code is organized into directories based on specific projects, including MNIST classification, attention mechanisms, and a GPT-style language model.

## Directory Overview

### 1. **MNIST**
- **Description**: This directory contains a simple implementation of a classifier for the MNIST dataset. The code demonstrates basic image classification techniques using PyTorch.
- **Key Features**:
  - Dataset loading and preprocessing.
  - Model definition with fully connected layers.
  - Training and evaluation scripts.

### 2. **Attention**
- **Description**: This directory implements attention mechanisms from scratch. It includes single-headed and multi-headed self-attention, explaining the core concepts behind transformers.
- **Key Features**:
  - Custom attention layer implementations.
  - Visualization of attention weights.
  - Step-by-step demonstration of how attention is computed.

### 3. **GPT**
- **Description**: This directory contains a Jupyter notebook that defines and trains a full language model based on the GPT architecture. The model is trained to generate text in the style of Shakespeare.
- **Key Features**:
  - Full GPT model implementation using PyTorch.
  - Training loop to fine-tune the model on Shakespearean text.
  - Sample text generation to mimic Shakespearean writing style.

## Usage

To use the code in this repository, clone the repository and navigate to the desired directory:

```bash
git clone <repository_url>
cd <directory_name>
```