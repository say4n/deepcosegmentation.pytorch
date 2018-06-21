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
OUTPUT_CHANNELS = 2 # BG + FG channel

## Training
CUDA = args.gpu
CHECKPOINT = args.checkpoint_save_dir
LOAD_CHECKPOOINT = args.checkpoint_load_dir
NUM_EPOCHS = 1


def train():
    is_better = True
    prev_loss = float('inf')

    model.train()

    for epoch in range(NUM_EPOCHS):
        loss_f = 0
        t_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"]
            labels = batch["label"]
            masks  = batch["mask"]

            pairwise_images = FloatTensor([(images[2*idx], images[2*idx+1]) for idx in range(BATCH_SIZE//2)])
            pairwise_labels = LongTensor([(labels[2*idx], labels[2*idx+1]) for idx in range(BATCH_SIZE//2)])
            pairwise_masks  = FloatTensor([(masks[2*idx], masks[2*idx+1]) for idx in range(BATCH_SIZE//2)])

            imagesA, imagesB = zip(*pairwise_images)
            labelsA, labelsB = zip(*pairwise_labels)
            masksA, masksB = zip(*pairwise_masks)

            pmapA, pmapB = model(imagesA, imagesB)

            optimizer.zero_grad()

            lossA = loss(pmapA, masksA)
            lossB = loss(pmapB, masksB)

            loss = lossA + lossB

            loss.backward()

            optimizer.step()


            # Add losses for epoch
            loss_f += loss.float()

        delta = time.time() - t_start
        is_better = loss_f < prev_loss


        if is_better:
            prev_loss = loss_f

            if CHECKPOINT:
                torch.save(model.state_dict(), os.path.join(CHECKPOINT, "model_best.pth"))

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, delta))




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

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=LEARNING_RATE, betas=BETAS)

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

    if CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        model.cuda()
        loss.cuda()

        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor

    if LOAD_CHECKPOOINT:
        model.load_state_dict(torch.load(LOAD_CHECKPOINT))


    train()
