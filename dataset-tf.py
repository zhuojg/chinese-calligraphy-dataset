import tensorflow as tf
import os
import pathlib
import numpy as np


class CalligraphyDataset:
    def _process_path(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize_with_pad(img, target_height=140, target_width=140)
        
        character = tf.strings.split(file_path, os.sep)[-2]
        
        return img, character

    def _get_embedding(self, character_csv):
        characters = {}

        with open(character_csv, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                characters[line.split(',')[0]] = int(line.split(',')[1])

        return characters


    def __init__(self, data_dir, character_csv, batch_size=1, repeat=True, shuffle=True, shuffle_buffer_size=32):
        data_dir = pathlib.Path(data_dir)
        list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))

        self.length = len(list_ds)
        self.class_num = len(os.listdir(data_dir))
        print('Found %d images in %d classes.' % (self.length, self.class_num))

        labeled_ds = list_ds.map(self._process_path)

        dataset = labeled_ds

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size=batch_size)
        
        self.dataset = dataset.as_numpy_iterator()

        self.characters = self._get_embedding(character_csv)

    def __len__(self):
        return self.length

    def next(self):
        (img, characters) = self.dataset.next()

        characters = np.array([self.characters[item.decode('utf-8')] for item in characters])

        return (img, characters)
        

if __name__ == '__main__':
    data = CalligraphyDataset(data_dir='./data/character/', character_csv='./data/label_character.csv', batch_size=8, repeat=False, shuffle=False)
    print(data.next())
