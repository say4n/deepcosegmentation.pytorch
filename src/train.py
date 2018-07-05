"""
Train SegNet based Siamese network

usage: train.py --dataset_root /home/SharedData/intern_sayan/PASCAL_coseg/ \
                --img_dir images \
                --mask_dir GT \
                --checkpoint_save_dir /home/SharedData/intern_sayan/PASCAL_coseg/ \
                --gpu 0
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
BATCH_SIZE = 2 * 2  # two images at a time for Siamese net - 2x batch_size hence
INPUT_CHANNELS = 3  # RGB
OUTPUT_CHANNELS = 2 # BG + FG channel

## Training
CUDA = args.gpu
CHECKPOINT = args.checkpoint_save_dir
LOAD_CHECKPOINT = args.checkpoint_load_dir
NUM_EPOCHS = 2000



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
            masksA, masksB = torch.stack(masksA).long(), torch.stack(masksB).long()

            # pdb.set_trace()

            eq_labels = []

            for idx in range(BATCH_SIZE//2):
                if torch.equal(labelsA[idx], labelsB[idx]):
                    eq_labels.append(torch.ones(1).type(LongTensor))
                else:
                    eq_labels.append(torch.zeros(1).type(LongTensor))

            eq_labels = torch.stack(eq_labels)

            masksA = masksA * eq_labels.unsqueeze(1).unsqueeze(1)
            masksB = masksB * eq_labels.unsqueeze(1).unsqueeze(1)

            # pdb.set_trace()

            imagesA_v = torch.autograd.Variable(FloatTensor(imagesA))
            imagesB_v = torch.autograd.Variable(FloatTensor(imagesB))

            pmapA, pmapB, similarity = model(imagesA_v, imagesB_v)

            # pdb.set_trace()

            optimizer.zero_grad()

            masksA_v = torch.autograd.Variable(LongTensor(masksA))
            masksB_v = torch.autograd.Variable(LongTensor(masksB))

            # pdb.set_trace()

            lossA = criterion(pmapA * similarity, masksA_v)
            lossB = criterion(pmapB * similarity, masksB_v)
            lossClasifier = classifier_criterion(similarity, eq_labels)

            loss = lossA + lossB + lossClasifier

            loss.backward()

            optimizer.step()


            # Add losses for epoch
            loss_f += loss.float()
            lossA_f += lossA.float()
            lossB_f += lossB.float()
            lossC_f += lossClasifier.float()


            # metrics - IoU & precision
            intersection_a, intersection_b, union_a, union_b, precision_a, precision_b = 0, 0, 0, 0, 0, 0

            for idx in range(BATCH_SIZE//2):
                pred_maskA = torch.argmax(pmapA[idx], dim=0).cpu().numpy()
                pred_maskB = torch.argmax(pmapB[idx], dim=0).cpu().numpy()

                masksA_cpu = masksA[idx].cpu().numpy()
                masksB_cpu = masksB[idx].cpu().numpy()

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
                torch.save(model.state_dict(), os.path.join(CHECKPOINT, "coseg_model_best.pth"))

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, delta))


if __name__ == "__main__":
    root_dir = args.dataset_root

    image_dir = os.path.join(root_dir, args.img_dir)
    mask_dir = os.path.join(root_dir, args.mask_dir)

    # iCoseg_dataset = iCosegDataset(image_dir=image_dir,
    #                                mask_dir=mask_dir)

    # dataloader = DataLoader(iCoseg_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

    PASCALVOCCoseg_dataset = PASCALVOCCosegDataset(image_dir=image_dir,
                                   mask_dir=mask_dir)

    dataloader = DataLoader(PASCALVOCCoseg_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

    #-------------#
    #    Model    #
    #-------------#

    model = SiameseSegNet(input_channels=INPUT_CHANNELS,
                          output_channels=OUTPUT_CHANNELS,
                          gpu=CUDA)

    if DEBUG:
        print(model)

    criterion = nn.CrossEntropyLoss()
    classifier_criterion = nn.CrossEntropyLoss()
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
        classifier_criterion = classifier_criterion.cuda()

        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor

    if LOAD_CHECKPOINT:
        model.load_state_dict(torch.load(os.path.join(LOAD_CHECKPOINT, "coseg_model_best.pth")))


    #-------------#
    #    Train    #
    #-------------#

    writer = SummaryWriter()

    train()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
