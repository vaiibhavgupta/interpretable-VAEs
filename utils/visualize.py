import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_tsne_clustering(model, data_loader, testing_in_training=False, epoch=None):
    DEVICE = torch.device('mps')
    model.eval()
    model.to(DEVICE)

    source_loader = data_loader['MNIST']
    source_latent_vectors, source_labels_list = [], []
    with torch.no_grad():
        for data, labels in source_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            _, _, _, _, mu_y, _, _, _ = model(data, labels)
            source_latent_vectors.append(mu_y.cpu().numpy())
            source_labels_list.append(np.array([0] * len(labels)))

    source_latent_vectors = np.concatenate(source_latent_vectors, axis=0)
    source_labels_list = np.concatenate(source_labels_list, axis=0)

    target_dataset = [key for key in data_loader.keys() if key != 'MNIST'][0]
    target_loader = data_loader[target_dataset]
    target_latent_vectors, target_labels_list = [], []
    with torch.no_grad():
        for data, labels in target_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            _, _, _, _, mu_y, _, _, _ = model(data)
            target_latent_vectors.append(mu_y.cpu().numpy())
            # print('LABEL', labels.cpu().numpy())
            target_labels_list.append(np.array([1] * len(labels)))

    target_latent_vectors = np.concatenate(target_latent_vectors, axis=0)
    target_labels_list = np.concatenate(target_labels_list, axis=0)

    latent_vectors = np.concatenate([source_latent_vectors, target_latent_vectors], axis=0)
    label_list = np.concatenate([source_labels_list, target_labels_list], axis=0)

    tsne = TSNE(n_components=2, perplexity=40, n_iter=500, learning_rate='auto', init='random')
    tsne_result = tsne.fit_transform(latent_vectors)

    fig = plt.figure(figsize=(15, 6))
    scatter1 = plt.scatter(
        tsne_result[:, 0], tsne_result[:, 1], 
        c=label_list, cmap='tab10', alpha=0.7
    )
    plt.title("t-SNE on Domain Separation")
    # fig.colorbar(scatter1, ax=axes[0], ticks=range(10))

    # plt.suptitle(f"t-SNE Visualization of Latent Space Alignment: MNIST to {target_dataset}")
    if testing_in_training:
        plt.savefig(f't-SNE (Epoch {epoch}) - MNIST to {target_dataset}.png')
    else:
        plt.savefig(f't-SNE - MNIST to {target_dataset}.png')

def visualize_reconstruction(model, data_loader, testing_in_training=True, epoch=None):
    DEVICE = torch.device('mps')
    model.eval()
    model.to(DEVICE)

    source_loader = data_loader['MNIST']
    target_dataset = [key for key in data_loader.keys() if key != 'MNIST'][0]
    target_loader = data_loader[target_dataset]

    def plot_images(original, reconstructed, labels, dataset_name):
        labels_to_collect = {k: None for k in range(10)}
        for idx, label in enumerate(labels):
            label = label.item()
            if labels_to_collect[label] is None:
                labels_to_collect[label] = idx

        _, axes = plt.subplots(2, 10, figsize=(15, 6))
        axes[0, 0].set_ylabel('Original', fontsize=16, rotation=0, labelpad=70)
        axes[1, 0].set_ylabel('Reconstructed', fontsize=16, rotation=0, labelpad=70)

        for i in range(10):
            axes[0, i].imshow(original[labels_to_collect[i]].squeeze().cpu().numpy(), cmap='gray')
            # axes[0, i].set_title(f"Original  Images - {dataset_name}")
            axes[0, i].axis('off')

            axes[1, i].imshow(reconstructed[labels_to_collect[i]].squeeze().cpu().numpy(), cmap='gray')
            # axes[1, i].set_title(f"Reconstructed Images - {dataset_name}")
            axes[1, i].axis('off')

        if testing_in_training:
            plt.savefig(f'Image Reconstruction (Epoch {epoch}) - {dataset_name}.png')
        else:
            plt.savefig(f'Image Reconstruction - {dataset_name}.png')

    with torch.no_grad():
        for source_data, source_labels in source_loader:
            source_data, source_labels = source_data.to(DEVICE), source_labels.to(DEVICE)
            source_recon, _, _, _, _, _, _, _ = model(source_data, source_labels)

        for target_data, target_labels in target_loader:
            target_data = target_data.to(DEVICE)
            target_recon, _, _, _, _, _, _, _ = model(target_data)

    plot_images(source_data, source_recon, source_labels, 'MNIST')
    plot_images(target_data, target_recon, target_labels, target_dataset)

def visualize_losses(total_losses, vae_losses, classification_losses, domain_losses, entropy_losses, target_dataset):
    vae_means = [torch.mean(torch.stack(epoch)).item() for epoch in vae_losses]
    classification_means = [torch.mean(torch.stack(epoch)).item() for epoch in classification_losses]
    domain_means = [torch.mean(torch.stack(epoch)).item() for epoch in domain_losses]
    entropy_losses = [torch.mean(torch.stack(epoch)).item() for epoch in entropy_losses]

    fig, axs = plt.subplots(5, 1, figsize=(12, 18))

    axs[0].plot(vae_means, marker='o')
    axs[0].set_title('VAE Loss Trend over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('VAE Loss')
    axs[0].grid(True)

    axs[1].plot(classification_means, marker='o')
    axs[1].set_title('Classification Loss (Source) Trend over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Classification Loss')
    axs[1].grid(True)

    axs[2].plot(domain_means, marker='o')
    axs[2].set_title('Domain Loss Trend over Epochs')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Domain Loss')
    axs[2].grid(True)

    axs[3].plot(entropy_losses, marker='o')
    axs[3].set_title('Entropy Loss (Target) Trend over Epochs')
    axs[3].set_xlabel('Epochs')
    axs[3].set_ylabel('Entropy Loss')
    axs[3].grid(True)

    axs[4].plot(total_losses, marker='o')
    axs[4].set_title('Total Loss Trend over Epochs')
    axs[4].set_xlabel('Epochs')
    axs[4].set_ylabel('Total Loss')
    axs[4].grid(True)

    plt.tight_layout()
    plt.savefig(f'Training Loss - MNIST to {target_dataset}.png')