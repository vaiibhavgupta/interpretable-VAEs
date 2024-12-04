from models.classifier import LabelClassifier, DomainClassifier
from models.encoder import Encoder
from models.decoder import Decoder
from models.grl import GRL

import math
import torch
import torch.nn as nn

class DisentangledVAE(nn.Module):
    def __init__(self, latent_dim_y, latent_dim_d):
        super(DisentangledVAE, self).__init__()
        self.encoder = Encoder(latent_dim_y, latent_dim_d)
        
        self.decoder_y = Decoder(latent_dim_y)
        self.decoder_d = Decoder(latent_dim_d)

        self.label_classifier = LabelClassifier(latent_dim_y, 10)
        self.domain_classifier = DomainClassifier(latent_dim_d, 2)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_lambda(self, epoch, type):
        # progress = epoch / self.lambda_steps
        # if type == 'exponential':
        #     return min(self.lambda_start + (self.lambda_end - self.lambda_start) * (1 - math.exp(-0.1 * progress)), self.lambda_end)
        # elif type == 'cyclical':
        #     cycle_length = self.lambda_steps
        #     return self.lambda_start + (self.lambda_end - self.lambda_start) * (0.5 * (1 + math.cos((epoch % cycle_length) / cycle_length * math.pi)))
        if type == 'fixed':
            return 0.5
        else:
            raise ValueError("type must be `exponential` or `cyclical` or `fixed`")

    def forward(self, x, labels=None, epoch=None, reverse_grad=False):
        mu_y, logvar_y, mu_d, logvar_d = self.encoder(x)
        z_y = self.reparameterize(mu_y, logvar_y)
        z_d = self.reparameterize(mu_d, logvar_d)

        # if labels is not None:
        #     x_recon = self.decoder_source(z_y)
        #     label_logits = self.label_classifier(z_y)
        # else:
        #     x_recon = self.decoder_target(z_y)
        #     label_logits = self.label_classifier(z_y)

        x_recon_y = self.decoder_y(z_y)
        x_recon_d = self.decoder_d(z_d)

        label_logits = self.label_classifier(z_y)

        if reverse_grad and epoch is not None:
            lambda_y = self.compute_lambda(epoch, "fixed")
            lambda_d = self.compute_lambda(epoch, "fixed")
            z_y = GRL.apply(z_y, lambda_y)
            z_d = GRL.apply(z_d, lambda_d)

        domain_logits = self.domain_classifier(z_d)
        return x_recon_y, x_recon_d, mu_y, logvar_y, mu_d, logvar_d, label_logits, domain_logits