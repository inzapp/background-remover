import cv2
import numpy as np
import tensorflow as tf
from random import shuffle
from concurrent.futures.thread import ThreadPoolExecutor


class AAEDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, input_shape, batch_size, add_noise=False, vertical_shake_power=0, horizontal_shake_power=0):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.add_noise = add_noise
        self.vertical_shake_power = vertical_shake_power
        self.horizontal_shake_power = horizontal_shake_power
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
            if self.add_noise:
                img = self.random_adjust(img)
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

    def random_adjust(self, img):
        if np.random.uniform() > 0.5:
            return img

        # adjust_opts = ['contrast', 'noise', 'motion_blur', 'loss']
        adjust_opts = ['motion_blur']
        # shuffle(adjust_opts)
        for i in range(len(adjust_opts)):
            img = self.adjust(img, adjust_opts[i])
        return img

    def adjust(self, img, adjust_type):
        if self.input_shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        weight = np.random.uniform(0.75, 1.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)

        if adjust_type == 'hue':
            h = np.asarray(h).astype('float32') * weight
            h = np.clip(h, 0.0, 255.0).astype('uint8')
        elif adjust_type == 'saturation':
            s = np.asarray(s).astype('float32') * weight
            s = np.clip(s, 0.0, 255.0).astype('uint8')
        elif adjust_type == 'brightness':
            v = np.asarray(v).astype('float32') * weight
            v = np.clip(v, 0.0, 255.0).astype('uint8')
        elif adjust_type == 'contrast':
            weight = np.random.uniform(0.0, 0.5)
            criteria = np.random.uniform(84.0, 170.0)
            v = np.asarray(v).astype('float32')
            v += (criteria - v) * weight
            v = np.clip(v, 0.0, 255.0).astype('uint8')
        elif adjust_type == 'noise':
            range_min = np.random.uniform(0.0, 100.0)
            range_max = np.random.uniform(0.0, 100.0)
            v = np.asarray(v).astype('float32')
            v += np.random.uniform(-range_min, range_max, size=v.shape)
            v = np.clip(v, 0.0, 255.0).astype('uint8')
        elif adjust_type == 'loss':
            origin_height, origin_width = img.shape[0], img.shape[1]
            fx = np.random.uniform(0.33, 1.0)
            fy = np.random.uniform(0.33, 1.0)
            img = cv2.merge([h, s, v])
            img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(img, (origin_width, origin_height), interpolation=cv2.INTER_AREA)
            h, s, v = cv2.split(img)
        elif adjust_type == 'motion_blur' and (self.vertical_shake_power > 1 or self.horizontal_shake_power > 1):
            size = None
            kernel = None
            if self.vertical_shake_power > 1 and self.horizontal_shake_power > 1:
                if np.random.uniform() > 0.5:
                    size = np.random.randint(1, self.vertical_shake_power)
                    kernel = np.zeros((size, size))
                    kernel[:, int((size - 1) / 2)] = np.ones(size)
                else:
                    size = np.random.randint(1, self.horizontal_shake_power)
                    kernel = np.zeros((size, size))
                    kernel[int((size - 1) / 2), :] = np.ones(size)
            elif self.vertical_shake_power > 1:
                size = np.random.randint(1, self.vertical_shake_power)
                kernel = np.zeros((size, size))
                kernel[:, int((size - 1) / 2)] = np.ones(size)
            elif self.horizontal_shake_power > 1:
                size = np.random.randint(1, self.horizontal_shake_power)
                kernel = np.zeros((size, size))
                kernel[int((size - 1) / 2), :] = np.ones(size)

            kernel /= size
            img = cv2.merge([h, s, v])
            img = cv2.filter2D(img, -1, kernel)
            h, s, v = cv2.split(img)

        img = cv2.merge([h, s, v])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        if self.input_shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
