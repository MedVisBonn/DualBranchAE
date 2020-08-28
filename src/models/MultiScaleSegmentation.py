import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class Model(nn.Module):
    def __init__(self, num_classes, width=145, pad=14):
        super().__init__()

        self.width = width
        self.pad = pad

        self.encoder_local_net = nn.Sequential(
            nn.Conv2d(288, 88, 1, stride=1, padding=0, bias=True),
            nn.PReLU(88),

            nn.Conv2d(88, 44, 1, stride=1, padding=0, bias=True),
            nn.PReLU(44),

            nn.Conv2d(44, 22, 1, stride=1, padding=0, bias=True),
            nn.PReLU(22),

            nn.Conv2d(22, 22, 1, stride=1, padding=0, bias=True),
            nn.PReLU(22),

            nn.Conv2d(22, 22, 1, stride=1, padding=0, bias=True),
            nn.PReLU(22)
        )

        self.encoder_regional_net = nn.Sequential(
            nn.Conv2d(288, 88, 5, stride=2, padding=0, bias=True),
            nn.PReLU(88),

            nn.Conv2d(88, 44, 5, stride=2, padding=0, bias=True),
            nn.PReLU(44),

            nn.Conv2d(44, 22, 5, stride=2, padding=0, bias=True),
            nn.PReLU(22),

            nn.Conv2d(22, 22, 5, stride=2, padding=0, bias=True),
            nn.PReLU(22),

            nn.Conv2d(22, 22, 5, stride=2, padding=0, bias=True),
            nn.PReLU(22)
        )

        self.classifier_net = nn.Sequential(
            nn.Conv2d(44, 88, 5, stride=1, padding=2, bias=True),
            nn.PReLU(88),

            nn.Conv2d(88, num_classes, 5, stride=1, padding=2, bias=True)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        crop = x[:, :, self.pad:self.pad + self.width, self.pad:self.pad + self.width]
        local_features = self.encoder_local_net(crop)
        regional_features = self.encoder_regional_net(x)
        # regional_features = interpolate(regional_features, size=(self.width, self.width),
        #                                 mode="bilinear", align_corners=True)

        regional_features = interpolate(regional_features, size=(161, 161),
                                        mode="bilinear", align_corners=True)
        regional_features = regional_features[:, :, 8:8 + 145, 8:8 + 145]

        encoded = torch.cat([local_features, regional_features], dim=1)

        return encoded

    def classify(self, encoded):
        return self.classifier_net(encoded)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.classify(encoded)
        return encoded, decoded
