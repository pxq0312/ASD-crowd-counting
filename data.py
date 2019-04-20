import torch.utils.data as data
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import cv2
import h5py

part = 'part_B'
out_scale = 16
root = '/media/disk1/pxq/ShanghaiTech/'
rootpath = os.path.join(root, part, 'train_data', 'images')


class TrainDataset(data.Dataset):
    def __init__(self):
        self.image_root = rootpath
        self.density_root = rootpath.replace('images', 'density_map')
        self.image_names = os.listdir(self.image_root)
        self.image_names.sort()
        self.density_names = os.listdir(self.density_root)
        self.density_names.sort()

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image_names[index])
        density_path = os.path.join(self.density_root, self.density_names[index])

        image = Image.open(image_path).convert('RGB')
        density = h5py.File(density_path, 'r')
        density = np.array(density['density'], dtype=np.float32)

        height, width = density.shape[0], density.shape[1]

        scale = random.uniform(0.8, 1.2)
        height, width = int(height * scale), int(width * scale)
        scale_transforms = transforms.Resize((height, width))
        image = scale_transforms(image)
        density = cv2.resize(density, (width, height)) / scale / scale

        h, w = 400, 400
        dh = random.randrange(0, height - h)
        dw = random.randrange(0, width - w)
        image = image.crop((dw, dh, dw + w, dh + h))
        density = density[dh:dh + h, dw:dw + w]

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            density = density[:, ::-1].copy()

        density = cv2.resize(density, (h // out_scale, w // out_scale)) * out_scale * out_scale

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        if random.random() < 0.3:
            gamma = random.uniform(0.5, 1.5)
            image = image ** gamma

        data_transforms = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return data_transforms(image), np.reshape(density, [1, density.shape[0], density.shape[1]])

    def __len__(self):
        return len(self.image_names)


class TestDataset(data.Dataset):
    def __init__(self):
        self.image_root = rootpath.replace('train', 'test')
        self.density_root = rootpath.replace('train', 'test').replace('images', 'density_map')
        self.image_names = os.listdir(self.image_root)
        self.image_names.sort()
        self.density_names = os.listdir(self.density_root)
        self.density_names.sort()

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image_names[index])
        density_path = os.path.join(self.density_root, self.density_names[index])

        image = Image.open(image_path).convert('RGB')
        density = h5py.File(density_path, 'r')
        density = np.array(density['density'], dtype=np.float32)

        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return data_transforms(image), np.reshape(density, [1, density.shape[0], density.shape[1]])

    def __len__(self):
        return len(self.image_names)


if __name__ == '__main__':
    train_dataset = TrainDataset()
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for image, label in train_loader:
        print(image.size())
        print(label.size())
        print(label.sum().item())
        img = np.transpose(image.numpy().squeeze(), [1, 2, 0]) * 0.2 + 0.45
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(label.squeeze(), cmap='jet')
        plt.show()
