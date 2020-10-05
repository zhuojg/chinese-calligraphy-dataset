import tensorflow as tf
import os
import pathlib
import numpy as np


class CalligraphyDataset:
    def _process_path(self, file_path):
        # read images from file  
        # and then convert RGB to grayscale
        # for the images only have black and white
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.rgb_to_grayscale(img)

        # convert black pixels to white and white piexels to black
        # 
        # in the original calligraphy, the character is black
        # but we want pixels of character to contain information
        img = tf.cast(img, dtype=tf.int32)
        img = tf.math.abs(img - 255)
        img = tf.cast(img, dtype=tf.uint8)

        # finally we resize the images but keep the ratio
        #
        # the target size (140, 140) is from EDA (see eda.ipynb)
        # the biggest height or width of all the images is 140
        img = tf.image.resize_with_pad(
            img, target_height=140, target_width=140)

        # we get the images corresponding labels from the file path
        # the path is like '../丁/xxx.jpg
        character = tf.strings.split(file_path, os.sep)[-2]

        return img, character

    def __init__(self, data_dir, character_csv, batch_size=1, repeat=True, shuffle=True, shuffle_buffer_size=32):
        data_dir = pathlib.Path(data_dir)
        list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))

        self.length = len(list_ds)
        self.class_num = len(os.listdir(data_dir))
        print('Found %d images in %d classes.' % (self.length, self.class_num))

        labeled_ds = list_ds.map(self._process_path)

        dataset = labeled_ds

        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size=batch_size)

        # self.dataset = dataset.as_numpy_iterator()
        self.dataset = dataset

        # read embedding file from character_csv
        self.characters = {}
        with open(character_csv, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.characters[line.split(',')[0]] = int(line.split(',')[1])

    def __len__(self):
        return self.length


if __name__ == '__main__':
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
