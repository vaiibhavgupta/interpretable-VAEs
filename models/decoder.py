import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # Combine latent dimensions for input size
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1)
        )

    def forward(self, z):
        x = F.relu(self.fc(z)).view(z.size(0), 512, 2, 2)
        return torch.sigmoid(self.deconv(x))
