import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_loss

def maximum_entropy_loss(logits):
    p = torch.softmax(logits, dim=1)
    log_p = torch.log(p + 1e-10)
    return -torch.mean(torch.sum(p * log_p, dim=1))