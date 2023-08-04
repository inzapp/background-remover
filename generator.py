"""
Authors : inzapp

Github url : https://github.com/inzapp/background-remover

Copyright 2021 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import cv2
import os
import numpy as np
import tensorflow as tf

from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 image_paths,
                 input_shape,
                 batch_size,
                 add_noise=False,
                 background_type='black'):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.add_noise = add_noise
        self.background_type = background_type
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0
        assert self.background_type in ['blur', 'black', 'gray', 'white', 'log', 'ada', 'dark']

    def __getitem__(self, index):
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image, self.next_image_path()))
        batch_x, batch_y, batch_m = [], [], []
        for f in fs:
            img, path = f.result()
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            batch_y.append(img.reshape(self.input_shape))
            if self.add_noise:
                img = self.random_adjust(img)
            batch_x.append(img.reshape(self.input_shape))
            batch_m.append(self.make_obj_mask(path).reshape(self.input_shape))
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32') / 255.0
        batch_y = np.asarray(batch_y).reshape((self.batch_size,) + self.input_shape).astype('float32') / 255.0
        batch_m = np.asarray(batch_m).reshape((self.batch_size,) + self.input_shape).astype('float32') / 255.0
        return batch_x, batch_y, batch_m

    def make_obj_mask(self, path):
        label_path = f'{path[:-4]}.txt'
        if not (os.path.exists(label_path) and os.path.isfile(label_path)):
            print('\nlabel not found : [{label_path}]')
            return None
        with open(label_path, 'rt') as f:
            lines = f.readlines()
        mask = np.zeros(shape=self.input_shape, dtype=np.uint8)
        height, width = self.input_shape[:2]
        for line in lines:
            class_index, cx, cy, w, h = list(map(float, line.split()))
            x1 = cx - w * 0.5
            x2 = cx + w * 0.5
            y1 = cy - h * 0.5
            y2 = cy + h * 0.5
            x1 = int(x1 * width)
            x2 = int(x2 * width)
            y1 = int(y1 * height)
            y2 = int(y2 * height)
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=-1)
        return mask

    def next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def load_image(self, image_path):
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if self.input_shape[-1] == 1 else cv2.IMREAD_COLOR), image_path

    def random_adjust(self, img):
        if np.random.uniform() > 0.5:
            return img
        adjust_opts = ['noise', 'blur']
        np.random.shuffle(adjust_opts)
        for i in range(len(adjust_opts)):
            img = self.adjust(img, adjust_opts[i])
        return img

    def adjust(self, img, adjust_type):
        if adjust_type == 'noise':
            range_min = np.random.uniform(0.0, 50.0)
            range_max = np.random.uniform(0.0, 50.0)
            img = np.asarray(img).astype('float32')
            img += np.random.uniform(-range_min, range_max, size=img.shape)
            img = np.clip(img, 0.0, 255.0).astype('uint8')
        elif adjust_type == 'blur':
            if np.random.uniform() > 0.5:
                img = cv2.blur(img, (2, 2))
            else:
                img = cv2.GaussianBlur(img, (3, 3), sigmaX=0)
        return img

