"""
Train SegNet based Siamese network

usage: train.py --dataset_root /home/SharedData/intern_sayan/PASCAL_coseg/ \
                --img_dir images \
                --mask_dir GT \
                --checkpoint_save_dir /home/SharedData/intern_sayan/PASCAL_coseg/ \
                --checkpoint_name deepcoseg_model_best.pth \
                --gpu 0

author - Sayan Goswami
email  - sayan.goswami.106@gmail.com
"""

import argparse
from dataset import iCosegDataset, PASCALVOCCosegDataset
from model import SiameseSegNet
import numpy as np
import os
import pdb
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

#-----------#
# Arguments #
#-----------#

parser = argparse.ArgumentParser(description='Train a SegNet model')

parser.add_argument('--dataset_root', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--checkpoint_save_dir', default=False)
parser.add_argument('--checkpoint_load_dir', default=False)
parser.add_argument('--checkpoint_name', default="deepcoseg_model_best.pth")
parser.add_argument('--gpu', default=None)

args = parser.parse_args()

#-----------#
# Constants #
#-----------#

## Debug
DEBUG = False

## Optimiser
LEARNING_RATE = 1e-5
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.0005

## Dataset
BATCH_SIZE = 2 * 1  # two images at a time for Siamese net - 2x batch_size hence
INPUT_CHANNELS = 3  # RGB
OUTPUT_CHANNELS = 1

## Training
CUDA = args.gpu
CHECKPOINT = args.checkpoint_save_dir
LOAD_CHECKPOINT = args.checkpoint_load_dir
NUM_EPOCHS = 500



def train():
    is_better = True
    prev_loss = float('inf')

    model.train()

    for epoch in range(NUM_EPOCHS):
        loss_f, lossA_f, lossB_f, lossC_f, intersection, union, precision = 0, 0, 0, 0, 0, 0, 0
        t_start = time.time()

        for batch_idx, batch in tqdm(enumerate(dataloader)):
            images = batch["image"].type(FloatTensor)
            labels = batch["label"].type(LongTensor)
            masks  = batch["mask"].type(FloatTensor)

            # pdb.set_trace()

            pairwise_images = [(images[2*idx], images[2*idx+1]) for idx in range(BATCH_SIZE//2)]
            pairwise_labels = [(labels[2*idx], labels[2*idx+1]) for idx in range(BATCH_SIZE//2)]
            pairwise_masks  = [(masks[2*idx], masks[2*idx+1]) for idx in range(BATCH_SIZE//2)]

            # pdb.set_trace()

            imagesA, imagesB = zip(*pairwise_images)
            labelsA, labelsB = zip(*pairwise_labels)
            masksA, masksB = zip(*pairwise_masks)

            # pdb.set_trace()

            imagesA, imagesB = torch.stack(imagesA), torch.stack(imagesB)
            labelsA, labelsB = torch.stack(labelsA), torch.stack(labelsB)
            masksA, masksB = torch.stack(masksA), torch.stack(masksB)

            # pdb.set_trace()

            eq_labels = []

            for idx in range(BATCH_SIZE//2):
                if torch.equal(labelsA[idx], labelsB[idx]):
                    eq_labels.append(torch.ones(1).type(FloatTensor))
                else:
                    eq_labels.append(torch.zeros(1).type(FloatTensor))

            eq_labels = torch.stack(eq_labels)

            # pdb.set_trace()

            masksA = masksA * eq_labels
            masksB = masksB * eq_labels


            imagesA_v = torch.autograd.Variable(imagesA.type(FloatTensor))
            imagesB_v = torch.autograd.Variable(imagesB.type(FloatTensor))


            pmapA, pmapB, similarity = model(imagesA_v, imagesB_v)


            # squeeze channels
            pmapA_sq = pmapA.squeeze(1)
            pmapB_sq = pmapB.squeeze(1)

            # pdb.set_trace()


            optimizer.zero_grad()

            lossA = criterion(pmapA_sq * eq_labels, masksA) / 512 * 512
            lossB = criterion(pmapB_sq * eq_labels, masksB) / 512 * 512
            lossClasifier = criterion(similarity, eq_labels) / BATCH_SIZE

            loss = lossA + lossB + lossClasifier

            # pdb.set_trace()

            loss.backward()

            optimizer.step()


            # Add losses for epoch
            loss_f += loss.cpu().float()
            lossA_f += lossA.cpu().float()
            lossB_f += lossB.cpu().float()
            lossC_f += lossClasifier.cpu().float()

            # metrics - IoU & precision
            intersection_a, intersection_b, union_a, union_b, precision_a, precision_b = 0, 0, 0, 0, 0, 0

            for idx in range(BATCH_SIZE//2):
                pdb.set_trace()

                pred_maskA = np.uint64(pmapA_sq[idx].detach().cpu().numpy())
                pred_maskB = np.uint64(pmapB_sq[idx].detach().cpu().numpy())

                masksA_cpu = np.uint64(masksA[idx].cpu().numpy())
                masksB_cpu = np.uint64(masksB[idx].cpu().numpy())

                intersection_a += np.sum(pred_maskA & masksA_cpu)
                intersection_b += np.sum(pred_maskB & masksB_cpu)

                union_a += np.sum(pred_maskA | masksA_cpu)
                union_b += np.sum(pred_maskB | masksB_cpu)

                precision_a += np.sum(pred_maskA == masksA_cpu)
                precision_b += np.sum(pred_maskB == masksB_cpu)

            intersection += intersection_a + intersection_b
            union += union_a + union_b

            precision += (precision_a / (512 * 512)) + (precision_b / (512 * 512))


        delta = time.time() - t_start


        writer.add_scalar("loss/lossA", lossA_f, epoch)
        writer.add_scalar("loss/lossB", lossB_f, epoch)
        writer.add_scalar("loss/lossClassifier", lossC_f, epoch)
        writer.add_scalar("loss/loss", loss_f, epoch)

        writer.add_scalar("metrics/precision", precision/(len(dataloader) * BATCH_SIZE), epoch)
        writer.add_scalar("metrics/iou", intersection/union, epoch)


        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f

            if CHECKPOINT:
                torch.save(model.state_dict(), os.path.join(CHECKPOINT, args.checkpoint_name))

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, delta))


if __name__ == "__main__":
    root_dir = args.dataset_root

    image_dir = os.path.join(root_dir, args.img_dir)
    mask_dir = os.path.join(root_dir, args.mask_dir)

    PASCALVOCCoseg_dataset = PASCALVOCCosegDataset(image_dir=image_dir,
                                                   mask_dir=mask_dir)

    dataloader = DataLoader(PASCALVOCCoseg_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    #-------------#
    #    Model    #
    #-------------#

    model = SiameseSegNet(input_channels=INPUT_CHANNELS,
                          output_channels=OUTPUT_CHANNELS,
                          gpu=CUDA)

    if DEBUG:
        print(model)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 betas=BETAS,
                                 weight_decay=WEIGHT_DECAY)

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

    if CUDA is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        model = model.cuda()
        criterion = criterion.cuda()

        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor

    if LOAD_CHECKPOINT:
        model.load_state_dict(torch.load(os.path.join(LOAD_CHECKPOINT, args.checkpoint_name)))


    #-------------#
    #    Train    #
    #-------------#

    writer = SummaryWriter()

    train()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
