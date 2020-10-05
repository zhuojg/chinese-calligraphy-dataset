# Chinese Calligraphy Dataset  

## Introduction  

From http://163.20.160.14/, we crawled hundreds of thousands of Chinese characters pictures written by different calligraphers.  

You can download data from [here](https://drive.google.com/file/d/1LeLbQGhCFLYJakQIjioZh4D9bD2izBSN/view?usp=sharing).  

You can download data **with calligraphers infomation** from [here](https://drive.google.com/file/d/1XznQ_wCSU3QvxnT5W5LeCZw4uF92FOcU/view?usp=sharing).  

## Usage  

> We write dataloader for dataset without calligrapher infomation.  

- Download data using the links above.
- Unzip the dataset in the `data` folder, which is in the same directory as `dataset-tf.py` and `dataset-pytorch.py`. The directory tree should be:

```shell
├── ...
├── data
│   ├── chinese-calligraphy-dataset
│   │   ├── ㄚ
│   │   ├── 一
│   │   ├── 丁
│   │   ├── 七
│   │   ├── 万
│   │   └── ...
│   └── label_character.csv
├── dataset-pytorch.py
├── dataset-tf.py
└── ...
```

- TensorFlow 2.3  

> The same code is in the `dataset-tf.py`, you can run `python dataset-tf.py` to see the results.  

```python
from dataset-tf import CalligraphyDataset
import matplotlib.pyplot as plt

batch_size = 8
sample_batch_num = 4
dataset = CalligraphyDataset(data_dir='./data/chinese-calligraphy-dataset/',
                            character_csv='./data/label_character.csv',
                            batch_size=8,
                            repeat=False,
                            shuffle=False)

plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']

for i_batch, (images, labels) in enumerate(dataset.dataset):
    if i_batch >= sample_batch_num:
        break

    labels = np.array([dataset.characters[item.numpy().decode('utf-8')] for item in labels])
    images = images.numpy()
    print(i_batch, images.shape, labels.shape)

    for i in range(images.shape[0]):
        ax = plt.subplot(sample_batch_num, batch_size, i_batch * batch_size + i + 1)
        ax.axis('off')
        ax.set_title(list(dataset.characters.keys())[labels[i]])
        plt.imshow(images[i])

plt.show()
```

- Pytorch  

> The same code is in the `dataset-pytorch.py`, you can run `python dataset-pytorch.py` to see the results.  

```python
from dataset-pytorch import CalligraphyDataset, ToTensor
import torch
import matplotlib.pyplot as plt

batch_size = 8
sample_batch_num = 4

dataset = CalligraphyDataset(
    data_dir='./data/character/',
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
```
