"""
Train SegNet based Siamese network
"""

import argparse
from model import SiameseSegNet
import os
import time
import torch
import torch.nn
from dataset import PascalVOCDeepCoSegmentationDataloader

#-----------#
# Arguments #
#-----------#

parser = argparse.ArgumentParser(description='Train a SegNet model')

parser.add_argument('--dataset_root', required=True)
parser.add_argument('--segmentation_dataset_path', required=True)
parser.add_argument('--classlabel_dataset_path', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--checkpoint_save_dir', default=False)
parser.add_argument('--checkpoint_load_dir', default=False)
parser.add_argument('--gpu', default=None)

args = parser.parse_args()

#-----------#
# Constants #
#-----------#

## Optimiser
LEARNING_RATE = 0.0004
BETAS = (0.9, 0.999)

## Dataset
BATCH_SIZE = 2 * 32 # two images at a time for Siamese net
INPUT_CHANNELS = 3 # RGB
OUTPUT_CHANNELS = 1 # Binary mask

## Training
CUDA = args.gpu
CHECKPOINT = args.checkpoint_save_dir
LOAD_CHECKPOOINT = args.checkpoint_load_dir


def train():
    pass


if __name__ == "__main__":
    root_dir = args.dataset_root

    segmentation_dataset = os.path.join(root_dir, args.segmentation_dataset_path)
    classlabel_dataset = os.path.join(root_dir, args.classlabel_dataset_path)
    image_dir = os.path.join(root_dir, args.img_dir)
    mask_dir = os.path.join(root_dir, args.mask_dir)

    dataset = PascalVOCDeepCoSegmentationDataloader(segmentation_dataset=segmentation_dataset,
                                                    classlabel_dataset=classlabel_dataset,
                                                    image_dir=image_dir,
                                                    mask_dir=mask_dir)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)



    model = SiameseSegNet(input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS)
    loss = nn.BCELoss()
    optimiser = torch.optim.Adam(lr=LEARNING_RATE, betas=BETAS)

    if CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        model.cuda()
        loss.cuda()

    if LOAD_CHECKPOOINT:
        model.load_state_dict(torch.load(LOAD_CHECKPOOINT))


    train()
