"""
Pytorch implementation of Deep Co-segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

DEBUG = False


vgg16_dims = [
                    (64, 64, 'M'),                                # Stage - 1
                    (128, 128, 'M'),                              # Stage - 2
                    (256, 256, 256,'M'),                          # Stage - 3
                    (512, 512, 512, 'M'),                         # Stage - 4
                    (512, 512, 512, 'M')                          # Stage - 5
            ]

decoder_dims = [
                    ('U', 512, 512, 512),                         # Stage - 5
                    ('U', 512, 512, 512),                         # Stage - 4
                    ('U', 256, 256, 256),                         # Stage - 3
                    ('U', 128, 128),                              # Stage - 2
                    ('U', 64, 64)                                 # Stage - 1
                ]


class SiameseSegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.input_channels = input_channels        # RGB = 3
        self.output_channels = output_channels      # FG + BG = 2


        def encoder_blocks(input_channels, output_channels, batch_norm=True):
            layers = []
            layers.append(nn.Conv2d(input_channels,
                                    output_channels,
                                    kernel_size=3,
                                    padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(output_channels))
            layers.append(nn.ReLU())

            return layers


        def decoder_blocks(input_channels, output_channels, batch_norm=True):
            layers = []
            layers.append(nn.ConvTranspose2d(input_channels,
                                             output_channels,
                                             kernel_size=3,
                                             padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(output_channels))
            layers.append(nn.ReLU())

            return layers


        self.encoder = models.vgg16(pretrained=True).features
        self.encoder_l2 = nn.Sequential(*encoder_blocks(512, 1024),
                                        *encoder_blocks(1024, 1024))

        self.decoder = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     *decoder_blocks(1024 + 1024, 512),
                                     *decoder_blocks(512, 512),
                                     *decoder_blocks(512, 512),
                                     nn.Upsample(scale_factor=2, mode='nearest'),
                                     *decoder_blocks(512, 512),
                                     *decoder_blocks(512, 512),
                                     *decoder_blocks(512, 512),
                                     nn.Upsample(scale_factor=2, mode='nearest'),
                                     *decoder_blocks(512, 256),
                                     *decoder_blocks(256, 256),
                                     *decoder_blocks(256, 256),
                                     nn.Upsample(scale_factor=2, mode='nearest'),
                                     *decoder_blocks(256, 128),
                                     *decoder_blocks(128, 128),
                                     nn.Upsample(scale_factor=2, mode='nearest'),
                                     *decoder_blocks(128, 64),
                                     *decoder_blocks(64, self.output_channels),
                                     nn.Softmax())


    def compute_correlation(self, featureA, featureB):
        # TODO: Fix this!
        # https://github.com/pytorch/pytorch/issues/4073
        B = featureA.shape[0]
        W = featureA.shape[-1]
        H = featureA.shape[-2]

        D = 2*max(W, H)
        patch_size = 3

        fA = featureA.transpose(1, 2).transpose(2, 3)       # B, H, W, C
        fB = featureB.transpose(1, 2).transpose(2, 3)       # B, H, W, C

        cAB = torch.zeros((B, W, H, D**2))                  # B, H, W, D^2

        for b in range(B):
            for i in range(H):
                for j in range(W):
                    for dm in range(-patch_size//2, patch_size//2 + 1):
                        for dn in range(-patch_size//2, patch_size//2 + 1):
                            m, n, k = i + dm, j + dn, dn * D + dm

                            if 0 <= m < H and 0 <= n < W:
                                cAB[b, i, j] += fA[b, i, j] * fB[b, m, n]

        return cAB.transpose_(3, 2).transpose_(2, 1)        # B, D^2, H, W


    def forward(self, imageA, imageB):
        """
        Forward pass images through the network
        """
        featureA = self.encoder_l2(self.encoder(imageA))
        featureB = self.encoder_l2(self.encoder(imageB))

        if DEBUG:
            print(f"featureA.size(): {featureA.size()}")
            print(f"featureB.size(): {featureB.size()}")

        correlationAB = self.compute_correlation(featureA, featureB)
        correlationBA = self.compute_correlation(featureB, featureA)

        if DEBUG:
            print(f"correlationAB.size(): {correlationAB.size()}")
            print(f"correlationBA.size(): {correlationBA.size()}")

        correspondence_mapA = torch.cat([featureA, correlationAB], dim=1)
        correspondence_mapB = torch.cat([featureB, correlationBA], dim=1)

        if DEBUG:
            print(f"correspondence_mapA.size(): {correspondence_mapA.size()}")
            print(f"correspondence_mapB.size(): {correspondence_mapB.size()}")

        probability_mapA = self.decoder(correspondence_mapA)
        probability_mapB = self.decoder(correspondence_mapB)

        if DEBUG:
            print(f"probability_mapA.size(): {probability_mapA.size()}")
            print(f"probability_mapB.size(): {probability_mapB.size()}")

        return probability_mapA, probability_mapB


if __name__ == "__main__":
    model = SiameseSegNet(input_channels=3, output_channels=2)

    iA = torch.rand((1, 3, 512, 512))
    iB = torch.rand((1, 3, 512, 512))

    pmapA, pmapB = model(iA, iB)

    if DEBUG:
        print(f"pmapA.size(), pmapB.size() : {pmapA.size()}, {pmapB.size()}")
