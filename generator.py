"""
Authors : inzapp

Github url : https://github.com/inzapp/auto-encoder

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


class AAEDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 image_paths,
                 input_shape,
                 batch_size,
                 vertical_shake_power=0,
                 horizontal_shake_power=0,
                 add_noise=False,
                 remove_background=False,
                 remove_background_type='black'):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.vertical_shake_power = vertical_shake_power
        self.horizontal_shake_power = horizontal_shake_power
        self.add_noise = add_noise
        self.remove_background = remove_background
        self.remove_background_type = remove_background_type
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0
        if self.remove_background:
            assert self.remove_background_type in ['blur', 'black', 'log', 'ada', 'dark']

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        batch_mask = []
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image, self.next_image_path()))
        for f in fs:
            img, path = f.result()
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            raw = img.copy()
            if self.add_noise:
                img = self.random_adjust(img)
            if self.remove_background:
                img = self.remove_background_process(raw, path)
            x = np.asarray(img).reshape(self.input_shape)
            if self.remove_background:
                batch_x.append(raw)
                batch_y.append(x.reshape(-1))
                batch_mask.append(self.make_obj_mask(path))
            else:
                batch_x.append(x)
                batch_y.append(raw.reshape(-1))
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32') / 255.0
        batch_y = np.asarray(batch_y).reshape((self.batch_size, int(np.prod(self.input_shape)))).astype('float32') / 255.0
        if self.remove_background:
            batch_mask = np.asarray(batch_mask).reshape((self.batch_size, int(np.prod(self.input_shape)))).astype('float32') / 255.0
        return batch_x, batch_y, batch_mask

    def remove_background_process(self, img, img_path):
        label_path = f'{img_path[:-4]}.txt'
        if not (os.path.exists(label_path) and os.path.isfile(label_path)):
            print('\nlabel not found : [{label_path}]')
            return img

        raw = img.copy()
        background_removed = None
        height, width = raw.shape[:2]
        if self.remove_background_type == 'blur':
            background_removed = cv2.blur(img, (32, 32))
        elif self.remove_background_type == 'black':
            background_removed = np.zeros(shape=self.input_shape, dtype=np.uint8)
        elif self.remove_background_type == 'log':
            if self.input_shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
            laplacian = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=1)
            laplacian = laplacian / np.max(laplacian)
            laplacian = np.where(laplacian > 25.5 / 255.0, 1.0, 0.0)
            laplacian = np.clip(laplacian * 255.0, 0.0, 255.0).astype('uint8')
            background_removed = laplacian
            if self.input_shape[-1] == 3:
                background_removed = cv2.cvtColor(background_removed, cv2.COLOR_GRAY2BGR)
        elif self.remove_background_type == 'ada':
            if self.input_shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            background_removed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -3)
            if self.input_shape[-1] == 3:
                background_removed = cv2.cvtColor(background_removed, cv2.COLOR_GRAY2BGR)
        elif self.remove_background_type == 'dark':
            background_removed = np.asarray(img).astype('float32') * 0.3
            background_removed = np.clip(background_removed, 0.0, 255.0).astype('uint8')

        with open(label_path, 'rt') as f:
            lines = f.readlines()
        for line in lines:
            class_index, cx, cy, w, h = list(map(float, line.split()))
            x1 = cx - w * 0.5
            x2 = cx + w * 0.5
            y1 = cy - h * 0.5
            y2 = cy + h * 0.5
            x1, x2, y1, y2 = np.clip(np.array([x1, x2, y1, y2], dtype=np.float32), 0.0, 1.0)
            x1 = int(x1 * width)
            x2 = int(x2 * width)
            y1 = int(y1 * height)
            y2 = int(y2 * height)
            for i in range(y1, y2):
                for j in range(x1, x2):
                    background_removed[i][j] = raw[i][j]
        return background_removed

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
        
        adjust_opts = ['motion_blur', 'noise', 'loss']
        # adjust_opts = ['contrast', 'motion_blur', 'noise', 'loss']
        np.random.shuffle(adjust_opts)
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
