from utils.checkpoint import save_checkpoint
from utils.loss import vae_loss, maximum_entropy_loss
from utils.visualize import visualize_losses, visualize_reconstruction, visualize_tsne_clustering
from utils.evaluate import evaluate

import time
import itertools

import torch
import torch.nn.functional as F

def train(model, source_loader, target_loader, optimizer, epochs, target_dataset, source_test_loader, target_test_loader):
    DEVICE = torch.device('mps')
    model.to(DEVICE)
    model.train()

    total_losses, vae_losses, classification_losses, domain_losses, entropy_losses = [], [], [], [], []
    for epoch in range(epochs):
        print(f'Epoch [{epoch + 1}/{epochs}]', end=' | ')
        
        st = time.time()
        lambda_ = model.compute_lambda(epoch, type='fixed')
        beta_ = 0.5
        alpha_ = 5

        total_loss = 0
        epoch_vae_losses, epoch_classification_losses, epoch_domain_losses, epoch_entropy_losses = [], [], [], []
        
        for (source_data, source_labels), (target_data, _) in itertools.zip_longest(source_loader, target_loader, fillvalue=(None, None)):
            if source_data is None or target_data is None:
                continue

            source_data, source_labels = source_data.to(DEVICE), source_labels.to(DEVICE)
            target_data = target_data.to(DEVICE)

            optimizer.zero_grad()

            source_recon_y, source_recon_d, source_mu_y, source_logvar_y, source_mu_d, source_logvar_d, source_label_logits, _ = model(source_data, source_labels)
            target_recon_y, target_recon_d, target_mu_y, target_logvar_y, target_mu_d, target_logvar_d, _, target_domain_logits = model(target_data)

            # x_recon_y, x_recon_d, mu_y, logvar_y, mu_d, logvar_d, label_logits, domain_logits

            classification_loss = F.cross_entropy(source_label_logits, source_labels.view(-1))

            vae_loss_source_y = vae_loss(source_recon_y, source_data, source_mu_y, source_logvar_y)
            vae_loss_source_d = vae_loss(source_recon_d, source_data, source_mu_d, source_logvar_d)
            vae_loss_source = vae_loss_source_y + vae_loss_source_d

            vae_loss_target_y = vae_loss(target_recon_y, target_data, target_mu_y, target_logvar_y)
            vae_loss_target_d = vae_loss(target_recon_d, target_data, target_mu_d, target_logvar_d)
            vae_loss_target = vae_loss_target_y + vae_loss_target_d

            _, _, _, _, _, _, _, source_domain_logits = model(source_data, source_labels, reverse_grad=True)
            domain_logits = torch.cat([source_domain_logits, target_domain_logits], dim=0)
            domain_labels = torch.cat([torch.zeros(len(source_data)), torch.ones(len(target_data))]).long().to(DEVICE)
            domain_loss = F.cross_entropy(domain_logits, domain_labels)

            entropy_loss = maximum_entropy_loss(target_domain_logits)
            
            epoch_vae_losses.append(vae_loss_source + vae_loss_target)
            epoch_classification_losses.append(beta_ * classification_loss)
            epoch_domain_losses.append(lambda_ * domain_loss)
            epoch_entropy_losses.append(alpha_ * entropy_loss)

            loss = ( (vae_loss_source + vae_loss_target) + (lambda_ * domain_loss) + (beta_ * classification_loss) + (alpha_ * entropy_loss ) )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        total_losses.append(total_loss)
        vae_losses.append(epoch_vae_losses)
        classification_losses.append(epoch_classification_losses)
        domain_losses.append(epoch_domain_losses)
        entropy_losses.append(epoch_entropy_losses)

        evaluate(model, {'MNIST': source_test_loader, target_dataset: target_test_loader})
        visualize_reconstruction(model, {'MNIST': source_test_loader, target_dataset: target_test_loader}, testing_in_training=True, epoch=epoch)
        visualize_tsne_clustering(model, {'MNIST': source_test_loader, target_dataset: target_test_loader}, testing_in_training=True, epoch=epoch)

        print(f'Training Loss: {total_loss / len(source_loader):.2f} | Time Elapsed: {round(time.time() - st, 2)} seconds')
        save_checkpoint(model, optimizer, epoch, target_dataset)
        
    visualize_losses(total_losses, vae_losses, classification_losses, domain_losses, entropy_losses, target_dataset)