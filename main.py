import warnings
warnings.filterwarnings('ignore')

from models.vae import DisentangledVAE
from utils.visualize import visualize_reconstruction, visualize_tsne_clustering

from utils.evaluate import evaluate
from utils.train import train

import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cpu')

BATCH_SIZE = 64
LATENT_DIM_Y = 10
LATENT_DIM_D = 4
EPOCHS = 2

def load_data(target_dataset):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    source_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    source_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    if target_dataset == 'USPS':
        target_train = datasets.USPS(root='./data', train=True, transform=transform, download=True)
        target_test = datasets.USPS(root='./data', train=False, transform=transform, download=True)
    
    elif target_dataset == 'SVHN':
        target_train = datasets.SVHN(root='./data', split='train', transform=transform, download=True)
        target_test = datasets.SVHN(root='./data', split='test', transform=transform, download=True)
    else:
        raise ValueError('Target Dataset can either be SVHN or USPS')

    source_train_loader = DataLoader(source_train, batch_size=BATCH_SIZE, shuffle=True)
    source_test_loader = DataLoader(source_test, batch_size=BATCH_SIZE, shuffle=False)

    target_train_loader = DataLoader(target_train, batch_size=BATCH_SIZE, shuffle=True)
    target_test_loader = DataLoader(target_test, batch_size=BATCH_SIZE, shuffle=False)

    return source_train_loader, source_test_loader, target_train_loader, target_test_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Disentangled VAE for Domain Adaptation")
    parser.add_argument('--train', action='store_true', help="Train the model on MNIST")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the model on SVHN")
    parser.add_argument('--visualize', action='store_true', help="Visualize t-SNE latent space")
    parser.add_argument('--dataset', type=str, help="`USPS` or `SVHN")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    target_dataset = args.dataset.strip()
    source_train_loader, source_test_loader, target_train_loader, target_test_loader = load_data(target_dataset=target_dataset)

    model = DisentangledVAE(latent_dim_y=LATENT_DIM_Y, latent_dim_d=LATENT_DIM_D).to(device)
    optimizer = Adam(model.parameters(), lr=5e-4)

    if args.train:
        train(model, source_train_loader, target_train_loader, optimizer, EPOCHS, target_dataset, source_test_loader, target_test_loader)

    if args.evaluate:
        evaluate(model, {'MNIST': source_test_loader, target_dataset: target_test_loader})

    if args.visualize:
        visualize_reconstruction(model, {'MNIST': source_test_loader, target_dataset: target_test_loader})
        visualize_tsne_clustering(model, {'MNIST': source_test_loader, target_dataset: target_test_loader})