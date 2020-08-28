import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class Model(nn.Module):
    def __init__(self, width, pad):
        super().__init__()

        self.width = width
        self.pad = pad

        self.encoder_local_net = nn.Sequential(
            nn.Conv3d(288, 88, 1, stride=1, padding=0, bias=True),
            nn.PReLU(88),

            nn.Conv3d(88, 44, 1, stride=1, padding=0, bias=True),
            nn.PReLU(44),

            nn.Conv3d(44, 22, 1, stride=1, padding=0, bias=True),
            nn.PReLU(22),
        )

        self.encoder_regional_net = nn.Sequential(
            nn.Conv3d(288, 88, 5, stride=2, padding=0, bias=True),
            nn.PReLU(88),

            nn.Conv3d(88, 44, 5, stride=2, padding=0, bias=True),
            nn.PReLU(44),

            nn.Conv3d(44, 22, 5, stride=2, padding=0, bias=True),
            nn.PReLU(22),
        )

        self.decoder_net = nn.Sequential(
            # nn.Conv3d(44, 88, 5, stride=1, padding=2, bias=True),
            # nn.PReLU(88),
            #
            # nn.Conv3d(88, 288, 5, stride=1, padding=2, bias=True)
            nn.Conv3d(44, 88, 1, stride=1, padding=0, bias=True),
            nn.PReLU(88),

            nn.Conv3d(88, 288, 1, stride=1, padding=0, bias=True)
        )

    def encode(self, x):
        crop = x[:, :, self.pad:self.pad + self.width, self.pad:self.pad + self.width, self.pad:self.pad + self.width]
        local_features = self.encoder_local_net(crop)
        regional_features = self.encoder_regional_net(x)
        regional_features = interpolate(regional_features, size=(self.width, self.width, self.width),
                                        mode="trilinear", align_corners=True)

        encoded = torch.cat([local_features, regional_features], dim=1)

        return encoded

    def decode(self, encoded):
        return self.decoder_net(encoded)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded
