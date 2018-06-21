"""
Dataloader for Cosegmentation
"""

import glob
import numpy as np
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset

DEBUG = False


class PascalVOCDeepCoSegmentationDataloader(Dataset):
    def __init__(self, *, segmentation_dataset, classlabel_dataset, image_dir, mask_dir):
        self.segmentation_dataset_path = segmentation_dataset
        self.classlabel_dataset_path = classlabel_dataset
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_dims = (224, 224)

        self._load_image_classes()

        self._load_segmentation_dataset()

    def _load_image_classes(self):
        self.image_class_lookup = dict()
        self.classes = set()

        for file in glob.glob(os.path.join(self.classlabel_dataset_path, "*")):
            filename = file.split("/")[-1]
            try:
                filename, train_val = filename.split("_")
                train_val = train_val.split(".")[0]

                self.classes.add(filename)

                if train_val == "trainval":
                    with open(file, "rt") as in_file:
                        lines = in_file.read()
                        lines = lines.split("\n")[:-1]
                        lines = [line.split(" ")[0] for line in lines if line.split(" ")[-1] == "1"]

                    if DEBUG:
                        print(filename, len(lines))

                    for image in lines:
                        self.image_class_lookup[str(image)] = filename

            except ValueError:
                pass

        self.classes = list(sorted(self.classes)) + ["background"]

    def class_label_to_class_index(self, label):
        return self.classes.index(label)

    def class_index_to_class_label(self, index):
        return self.classes[index]

    def image_name_to_class_label(self, image):
        return self.image_class_lookup[str(image)]

    def _load_segmentation_dataset(self):
        with open(self.segmentation_dataset_path) as in_file:
            lines = in_file.read().split("\n")[:-1]

        self.images = [(image_name,
                        self.class_label_to_class_index(self.image_name_to_class_label(image_name))) for image_name in lines]

    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = np.transpose(raw_image.resize(self.image_dims), (2,1,0))
        imx_t = np.array(raw_image, dtype=np.float32)/255.0

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize(self.image_dims)
        imx_t = np.array(raw_image)
        # border
        imx_t[imx_t==255] = len(self.classes) - 1

        return imx_t

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_name, class_index = self.images[index]

        image_name = im_name + ".jpg"
        image_path = os.path.join(self.image_dir, image_name)

        mask_name = im_name + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        data = {
            "image": self.load_image(image_path),
            "mask": self.load_mask(mask_path),
            "label": class_index
        }

        return data


if __name__ == "__main__":
    root_dir = "/Users/Sayan/Desktop/Research/IIT B/Vision/datasets/VOCdevkit/VOC2007/"

    segmentation_dataset = os.path.join(root_dir, "ImageSets/Segmentation/trainval.txt")
    classlabel_dataset = os.path.join(root_dir, "ImageSets/Main")
    image_dir = os.path.join(root_dir, "JPEGImages")
    mask_dir = os.path.join(root_dir, "SegmentationClass")

    dataset = PascalVOCDeepCoSegmentationDataloader(segmentation_dataset=segmentation_dataset,
                                                    classlabel_dataset=classlabel_dataset,
                                                    image_dir=image_dir,
                                                    mask_dir=mask_dir)

    print(dataset[0])
