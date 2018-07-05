"""
Train SegNet based Siamese network

usage: infer.py --dataset_root /home/SharedData/intern_sayan/iCoseg/ \
                --img_dir images \
                --mask_dir ground_truth \
                --checkpoint_load_dir /home/SharedData/intern_sayan/iCoseg/ \
                --output_dir ./results \
                --gpu 1
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
import torchvision
from tqdm import tqdm

#-----------#
# Arguments #
#-----------#

parser = argparse.ArgumentParser(description='Train a SegNet model')

parser.add_argument('--dataset_root', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--checkpoint_load_dir', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--gpu', default=None)

args = parser.parse_args()

#-----------#
# Constants #
#-----------#

## Debug

DEBUG = False


## Dataset
BATCH_SIZE = 2 * 1 # two images at a time for Siamese net
INPUT_CHANNELS = 3 # RGB
OUTPUT_CHANNELS = 2 # BG + FG channel

## Inference
CUDA = args.gpu
LOAD_CHECKPOINT = args.checkpoint_load_dir

## Output Dir
OUTPUT_DIR = args.output_dir

os.system(f"rm -r {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def infer():
    model.eval()

    intersection, union, precision = 0, 0, 0
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

        # pdb.set_trace()

        masksA = masksA * eq_labels.unsqueeze(1)
        masksB = masksB * eq_labels.unsqueeze(1)

        imagesA_v = torch.autograd.Variable(FloatTensor(imagesA))
        imagesB_v = torch.autograd.Variable(FloatTensor(imagesB))

        pmapA, pmapB, similarity = model(imagesA_v, imagesB_v)

        # pdb.set_trace()

        res_images, res_masks, gt_masks = [], [], []

        for idx in range(BATCH_SIZE//2):
            res_images.append(imagesA[idx])
            res_images.append(imagesB[idx])

            res_masks.append(torch.argmax((pmapA * similarity.unsqueeze(2).unsqueeze(2))[idx], dim=0).reshape(1, 512, 512))
            res_masks.append(torch.argmax((pmapB * similarity.unsqueeze(2).unsqueeze(2))[idx], dim=0).reshape(1, 512, 512))

            gt_masks.append(masksA[idx].reshape(1, 512, 512))
            gt_masks.append(masksB[idx].reshape(1, 512, 512))

        # pdb.set_trace()

        images_T = torch.stack(res_images)
        masks_T = torch.stack(res_masks)
        gt_masks_T = torch.stack(gt_masks)

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

        # pdb.set_trace()

        torchvision.utils.save_image(images_T, os.path.join(OUTPUT_DIR, f"batch_{batch_idx}_images.png"), nrow=2)
        torchvision.utils.save_image(masks_T, os.path.join(OUTPUT_DIR, f"batch_{batch_idx}_masks.png"), nrow=2)
        torchvision.utils.save_image(gt_masks_T, os.path.join(OUTPUT_DIR, f"batch_{batch_idx}_gt_masks.png"), nrow=2)

    delta = time.time() - t_start

    print(f"\nTime elapsed: [{delta} secs]\nPrecision : [{precision/(len(dataloader) * BATCH_SIZE)}]\nIoU : [{intersection/union}]")


if __name__ == "__main__":
    root_dir = args.dataset_root

    image_dir = os.path.join(root_dir, args.img_dir)
    mask_dir = os.path.join(root_dir, args.mask_dir)

    iCoseg_dataset = iCosegDataset(image_dir=image_dir,
                                   mask_dir=mask_dir)

    dataloader = DataLoader(iCoseg_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

    # PASCALVOCCoseg_dataset = PASCALVOCCosegDataset(image_dir=image_dir,
    #                                mask_dir=mask_dir)

    # dataloader = DataLoader(PASCALVOCCoseg_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

    #-------------#
    #    Model    #
    #-------------#

    model = SiameseSegNet(input_channels=INPUT_CHANNELS,
                          output_channels=OUTPUT_CHANNELS,
                          gpu=CUDA)

    if DEBUG:
        print(model)

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

    if CUDA is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        model = model.cuda()

        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor

    if LOAD_CHECKPOINT:
        model.load_state_dict(torch.load(os.path.join(LOAD_CHECKPOINT, "coseg_model_best.pth")))


    #------------#
    #    Test    #
    #------------#

    infer()
