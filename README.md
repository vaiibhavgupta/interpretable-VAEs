# Disentangled Semantic Representation for Unsupervised Domain Adaptation

## Overview

This project explores unsupervised domain adaptation (UDA) using a disentangled variational autoencoder (VAE). The goal is to train a model on the MNIST dataset (source domain) and adapt it to the SVHN or USPS dataset (target domain). By disentangling semantic and domain-specific features through a dual adversarial network, the model aims to improve generalization across domains with significant visual differences.

## Project Structure

```
project_root/
├── data/                # Directory for downloaded datasets
├── models/              # Model components
│   ├── encoder.py       # Encoder network
│   ├── decoder.py       # Decoder network
│   ├── vae.py           # Disentangled VAE model
│   ├── classifier.py    # Label and domain classifiers
│   └── grl.py           # Gradient Reversal Layer (GRL)
├── utils/               # Utility functions
│   ├── train.py         # Training loop and hyperparameter tuning
│   ├── evaluate.py      # Evaluation functions
│   ├── visualize.py     # t-SNE visualization functions
│   └── checkpoint.py    # Save and load model functions
├── main.py              # Main script to run the project
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Setup

### Clone the Repository:

```
git clone <repository_url>
cd project_root
```

### Install Dependencies:

```
pip install -r requirements.txt
```

## How to Run the Project

Run: 

```
python main.py --train --evaluate --visualize --dataset USPS
```