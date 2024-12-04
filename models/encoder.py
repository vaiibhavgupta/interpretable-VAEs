import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim_y, latent_dim_d):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.fc_mu_y = nn.Linear(512 * 2 * 2, latent_dim_y)
        self.fc_logvar_y = nn.Linear(512 * 2 * 2, latent_dim_y)
        self.fc_mu_d = nn.Linear(512 * 2 * 2, latent_dim_d)
        self.fc_logvar_d = nn.Linear(512 * 2 * 2, latent_dim_d)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc_mu_y(x), self.fc_logvar_y(x), self.fc_mu_d(x), self.fc_logvar_d(x)