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
    def __init__(self, input_channels, output_channels, gpu=None):
        super().__init__()

        self.input_channels = input_channels        # RGB = 3
        self.output_channels = output_channels      # FG + BG = 2

        self.gpu = gpu


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


        self.encoder = models.vgg16_bn(pretrained=True).features
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
                                     nn.Softmax(dim=1))

        self.classifier = nn.Sequential(nn.Linear(2 * (1024 * 16 * 16), 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 16),
                                        nn.ReLU(),
                                        nn.Linear(16, 2),
                                        nn.Softmax(dim=1))


    def compute_correlation(self, featureA, featureB):
        pass


    def pearsonr(x, y):
        """
        Mimics `scipy.stats.pearsonr`

        Arguments
        ---------
        x : 1D torch.Tensor
        y : 1D torch.Tensor

        Returns
        -------
        r_val : float
            pearsonr correlation coefficient between x and y

        Scipy docs ref:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

        Scipy code ref:
            https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
        Example:
            >>> x = np.random.randn(100)
            >>> y = np.random.randn(100)
            >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
            >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
            >>> np.allclose(sp_corr, th_corr)

        Source: https://github.com/pytorch/pytorch/issues/1254
        """
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x.sub(mean_x)
        ym = y.sub(mean_y)

        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)

        r_val = r_num / r_den

        return r_val



    def concat_correlation(self, featureA, featureB):
        # hack (?)
        return featureB


    def forward(self, imageA, imageB):
        """
        Forward pass images through the network
        """
        vggA_feat = self.encoder(imageA)
        vggB_feat = self.encoder(imageB)

        if DEBUG:
            print(f"vggA_feat.size(): {vggA_feat.size()}")
            print(f"vggB_feat.size(): {vggB_feat.size()}")

        featureA = self.encoder_l2(vggA_feat)
        featureB = self.encoder_l2(vggB_feat)

        if DEBUG:
            print(f"featureA.size(): {featureA.size()}")
            print(f"featureB.size(): {featureB.size()}")

        similarity = self.classifier(torch.cat([featureA.view(featureA.size(0), -1), featureB.view(featureA.size(0), -1)]))

        if DEBUG:
            print(f"similarity: {similarity}")

        correlationAB = self.concat_correlation(featureA, featureB)
        correlationBA = self.concat_correlation(featureB, featureA)

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

        return probability_mapA, probability_mapB, similarity


if __name__ == "__main__":
    model = SiameseSegNet(input_channels=3, output_channels=2)

    iA = torch.rand((1, 3, 512, 512))
    iB = torch.rand((1, 3, 512, 512))

    pmapA, pmapB = model(iA, iB)

    if DEBUG:
        print(f"pmapA.size(), pmapB.size() : {pmapA.size()}, {pmapB.size()}")
