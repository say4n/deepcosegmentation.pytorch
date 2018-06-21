"""
Train SegNet based Siamese network
"""

import argparse
from model import SiameseSegNet
import os
import time
import torch
from dataset import PascalVOCDeepCoSegmentationDataloader
