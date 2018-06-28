"""iCoseg Dataset"""


import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.transforms import Compose, Resize, ToTensor


class iCosegDataset(Dataset):
    def __init__(self, *, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.img_size = (512, 512)

        self.image_data = DatasetFolder(root=image_dir,
                                        loader=self.image_loader,
                                        extensions=["jpg"],
                                        transform=Compose([Resize(self.img_size, interpolation=0), ToTensor()]))

        self.mask_data = DatasetFolder(root=mask_dir,
                                       loader=self.mask_loader,
                                       extensions=["png"],
                                       transform=Compose([Resize(self.img_size, interpolation=0), ToTensor()]))

        self.length = len(self.image_data)

    def image_loader(self, path):
        return Image.open(path)

    def mask_loader(self, path):
        return Image.open(path)

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
    dataset = iCosegDataset(image_dir="/Users/Sayan/Desktop/Research/IIT B/Vision/datasets/iCoseg/images",
                            mask_dir="/Users/Sayan/Desktop/Research/IIT B/Vision/datasets/iCoseg/ground_truth")

    print(dataset[0])
