import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(PatchNet, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder
        for feat in features:
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = feat

        # Decoder
        for feat in reversed(features):
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat, feat, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = feat

        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # simple encoder-decoder
        for enc in self.encoders:
            x = F.max_pool2d(enc(x), 2)
        for dec in self.decoders:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            x = dec(x)
        return self.final(x)

