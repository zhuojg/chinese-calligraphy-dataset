import glob
import torch
import os
from skimage import io, transform
import numpy as np
import cv2
import pathlib


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, character = sample['image'], sample['character']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'character': torch.from_numpy(character)}


class CalligraphyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, character_csv=None, target_size=140, transform=None):
        self.images_list = glob.glob(
            str(pathlib.Path(data_dir)/'**/*.jpg'), recursive=True)
        self.target_size = target_size
        self.transform = transform

        self.characters = {}
        with open(character_csv, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.characters[line.split(',')[0]] = int(line.split(',')[1])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images_list[idx]
        image = io.imread(img_path).astype(np.int)
        image = np.absolute(image - 255).astype(np.uint8)

        # resize image
        # we need to keep the ratio of the original image
        x, y = image.shape
        canva_size = max(self.target_size, x, y)

        # we need to put the image in the middle of the canva
        top = int((canva_size - x) / 2)
        left = int((canva_size - y) / 2)
        canva = np.zeros((canva_size, canva_size), dtype=np.uint8) * 255
        canva[top: top + x, left: left + y] = image
        image = cv2.resize(canva, (self.target_size, self.target_size))

        # add channel dimension
        image = image[:, :, np.newaxis]

        # get the embedding of the character
        character = img_path.split(os.sep)[-2]
        character = np.array(self.characters[character])
        character = character[np.newaxis]

        sample = {'image': image, 'character': character}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    batch_size = 8
    sample_batch_num = 4

    dataset = CalligraphyDataset(
        data_dir='./data/chinese-calligraphy-dataset/', 
        character_csv='./data/label_character.csv', 
        # transform=torchvision.transforms.Compose([ToTensor()])    # uncomment this line to transform numpy to tensor
    )

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei']
    for i_batch, sample_batched in enumerate(data_loader):
        if i_batch == sample_batch_num:
            break

        print(i_batch, sample_batched['image'].size(),
            sample_batched['character'].size())
        
        
        for i in range(len(sample_batched['image'])):
            ax = plt.subplot(sample_batch_num, batch_size, i_batch * batch_size + i + 1)
            ax.axis('off')
            ax.set_title(list(dataset.characters.keys())[int(sample_batched['character'][i][0])])
            plt.imshow(sample_batched['image'][i])
        
    plt.show()
