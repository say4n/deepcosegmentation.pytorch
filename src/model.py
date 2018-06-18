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


class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.input_channels = input_channels        # RGB = 3
        self.output_channels = output_channels      # FG + BG = 2


        def decoder_blocks(input_channels, output_channels, batch_norm=True):
            layers = []
            layers.append(nn.ConvTranspose2d(input_channels,
                                             output_channels,
                                             kernel_size=3,
                                             padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(output_channels))

            return layers
        

        self.encoder = models.vgg16(pretrained=True).features
        self.decoder = nn.Sequential(nn.MaxUnpool2d(kernel_size=2, stride=2),
                                     *decoder_blocks(961 + 512, 512),
                                     *decoder_blocks(512, 512),
                                     *decoder_blocks(512, 512),
                                     nn.MaxUnpool2d(kernel_size=2, stride=2),
                                     *decoder_blocks(512, 512),
                                     *decoder_blocks(512, 512),
                                     *decoder_blocks(512, 512),
                                     nn.MaxUnpool2d(kernel_size=2, stride=2),
                                     *decoder_blocks(512, 256),
                                     *decoder_blocks(256, 256),
                                     *decoder_blocks(256, 256),
                                     nn.MaxUnpool2d(kernel_size=2, stride=2),
                                     *decoder_blocks(256, 128),
                                     *decoder_blocks(128, 128),
                                     nn.MaxUnpool2d(kernel_size=2, stride=2),
                                     *decoder_blocks(128, 64),
                                     *decoder_blocks(64, self.output_channels))

    
    def compute_correlation(self, featureA, featureB):
        pass


    def forward(self, imageA, imageB):
        """
        Forward pass images through the network
        """
        featureA = self.encoder(imageA)
        featureB = self.encoder(imageB)

        correlationAB = compute_correlation(featureA, featureB)
        correlationBA = compute_correlation(featureB, featureA)

        correspondence_mapA = torch.cat([featureA, correlationAB])
        correspondence_mapB = torch.cat([featureB, correlationBA])

        probability_mapA = self.decoder(correspondence_mapA)
        probability_mapB = self.decoder(correspondence_mapB)

        return probability_mapA, probability_mapB


if __name__ == "__main__":
    model = SegNet(input_channels=3, output_channels=2)

    fA = torch.rand((4, 1024, 16, 16))
    fB = torch.rand((4, 1024, 16, 16))

    print(model.compute_correlation(fA, fB).size())