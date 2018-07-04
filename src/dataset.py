"""iCoseg Dataset"""


import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.transforms import ToTensor


class iCosegDataset(Dataset):
    def __init__(self, *, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.img_size = (512, 512)
        self.in_channels = 3

        self.image_data = DatasetFolder(root=image_dir,
                                        loader=self.image_loader,
                                        extensions=["jpg"],
                                        transform=ToTensor())

        self.mask_data = DatasetFolder(root=mask_dir,
                                       loader=self.mask_loader,
                                       extensions=["png"])

        self.length = len(self.image_data)

    def image_loader(self, path):
        img = Image.open(path).resize(self.img_size)
        img = np.array(img).astype(np.float32)/255.0

        return img

    def mask_loader(self, path):
        img = Image.open(path).resize(self.img_size)
        img = np.array(img).astype(np.uint8)

        return img

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image, im_label = self.image_data[index]
        mask, ma_label = self.mask_data[index]

        label = None

        if im_label == ma_label:
            label = im_label

        data = {
            "image": image,
            "mask" : mask,
            "label": label
        }

        return data


class PASCALVOCCosegDataset(Dataset):
    def __init__(self, *, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.img_size = (512, 512)
        self.in_channels = 3

        self.image_data = DatasetFolder(root=image_dir,
                                        loader=self.image_loader,
                                        extensions=["jpg"],
                                        transform=ToTensor())

        self.mask_data = DatasetFolder(root=mask_dir,
                                       loader=self.mask_loader,
                                       extensions=["png"])

        self.length = len(self.image_data)

    def image_loader(self, path):
        img = Image.open(path).resize(self.img_size)
        img = np.array(img).astype(np.float32)/255.0

        return img

    def mask_loader(self, path):
        img = Image.open(path).convert('L').resize(self.img_size)
        img = np.array(img).astype(np.uint8)

        return img

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image, im_label = self.image_data[index]
        mask, ma_label = self.mask_data[index]

        label = None

        if im_label == ma_label:
            label = im_label

        data = {
            "image": image,
            "mask" : mask,
            "label": label
        }

        return data


if __name__ == "__main__":
    iCoseg_dataset = iCosegDataset(image_dir="/Users/Sayan/Desktop/Research/IIT B/Vision/datasets/iCoseg/images",
                                   mask_dir="/Users/Sayan/Desktop/Research/IIT B/Vision/datasets/iCoseg/ground_truth")
    print(f"iCoseg_dataset: # samples = {len(iCoseg_dataset)}")

    PASCALVOCCoseg_dataset = PASCALVOCCosegDataset(image_dir="/Users/Sayan/Desktop/Research/IIT B/Vision/datasets/PASCAL_coseg/images",
                                                   mask_dir="/Users/Sayan/Desktop/Research/IIT B/Vision/datasets/PASCAL_coseg/GT")
    print(f"PASCALVOCCoseg_dataset: # samples = {len(PASCALVOCCoseg_dataset)}")
