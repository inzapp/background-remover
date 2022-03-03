import cv2
import numpy as np
import tensorflow as tf
from random import shuffle
from concurrent.futures.thread import ThreadPoolExecutor


class AAEDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, input_shape, batch_size):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.random_indexes = np.arange(len(self.image_paths))
        self.pool = ThreadPoolExecutor(8)
        np.random.shuffle(self.random_indexes)

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        start_index = index * self.batch_size
        fs = []
        for i in range(start_index, start_index + self.batch_size):
            cur_img_path = self.image_paths[self.random_indexes[i]]
            fs.append(self.pool.submit(self.load_image, cur_img_path))
        for f in fs:
            img = f.result()
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            x = np.asarray(img).reshape(self.input_shape)
            batch_x.append(x)
            batch_y.append(x)
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32') / 255.0
        batch_y = np.asarray(batch_y).reshape((self.batch_size,) + self.input_shape).astype('float32') / 255.0
        return batch_x, batch_y

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def load_image(self, image_path):
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if self.input_shape[-1] == 1 else cv2.IMREAD_COLOR)

    def shuffle(self):
        np.random.shuffle(self.random_indexes)
