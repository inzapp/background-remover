"""
Authors : inzapp

Github url : https://github.com/inzapp/aae

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
import natsort
import numpy as np
import os
import random
import tensorflow as tf
from cv2 import cv2
from generator import AAEDataGenerator
from glob import glob
from model import Model
from time import time


class AutoEncoder:
    def __init__(self,
                 train_image_path=None,
                 input_shape=(128, 128, 1),
                 lr=0.001,
                 momentum=0.9,
                 batch_size=32,
                 encoding_dim=128,
                 iterations=100000,
                 validation_split=0.2,
                 validation_image_path='',
                 checkpoint_path='checkpoints',
                 training_view=False,
                 pretrained_model_path='',
                 denoise=False,
                 vertical_shake_power=0,
                 horizontal_shake_power=0):
        self.iterations = iterations
        self.training_view = training_view
        self.live_view_previous_time = time()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.denoise = denoise
        self.view_flag = 1

        self.model = Model(input_shape=input_shape, lr=lr, momentum=momentum, encoding_dim=encoding_dim)
        if os.path.exists(pretrained_model_path) and os.path.isfile(pretrained_model_path):
            print(f'\npretrained model path : {[pretrained_model_path]}')
            self.ae, self.input_shape, self.encoding_dim = self.model.load(pretrained_model_path)
            print(f'input_shape : {self.input_shape}')
            print(f'encoding_dim : {self.encoding_dim}')
        else:
            self.ae = self.model.build()
        self.encoder = self.model.extract_encoder()
        self.decoder = self.model.extract_decoder()

        if validation_image_path != '':
            self.train_image_paths, _ = self.init_image_paths(train_image_path)
            self.validation_image_paths, _ = self.init_image_paths(validation_image_path)
        elif validation_split > 0.0:
            self.train_image_paths, self.validation_image_paths = self.init_image_paths(train_image_path, validation_split=validation_split)

        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.train_data_generator = AAEDataGenerator(
            image_paths=self.train_image_paths,
            input_shape=input_shape,
            batch_size=batch_size,
            vertical_shake_power=vertical_shake_power,
            horizontal_shake_power=horizontal_shake_power,
            add_noise=denoise)
        self.validation_data_generator = AAEDataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=input_shape,
            batch_size=batch_size,
            vertical_shake_power=vertical_shake_power,
            horizontal_shake_power=horizontal_shake_power,
            add_noise=denoise)

    def get_encoding_dim(self, ae):
        for layer in ae.layers:
            print(layer)
            if layer.name == 'encoder_output':
                print(layer)
                pass
        return None

    def fit(self):
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print(f'validate on {len(self.validation_image_paths)} samples.')
        print('start training')
        self.train()

    def train(self):
        iteration_count = 0
        while True:
            for ae_x, ae_y in self.train_data_generator:
                iteration_count += 1
                loss = self.ae.train_on_batch(ae_x, ae_y, return_dict=True)['loss']
                print(f'\r[iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='\t')
                if self.training_view:
                    self.training_view_function()
                if iteration_count % 10000 == 0:
                    loss = self.model.ae.evaluate(x=self.validation_data_generator)
                    self.model.save(self.checkpoint_path, iteration_count, loss)
                if iteration_count == self.iterations:
                    print('\n\ntrain end successfully')
                    return

    @staticmethod
    def init_image_paths(image_path, validation_split=0.0):
        all_image_paths = glob(f'{image_path}/**/*.jpg', recursive=True)
        random.shuffle(all_image_paths)
        num_train_images = int(len(all_image_paths) * (1.0 - validation_split))
        image_paths = all_image_paths[:num_train_images]
        validation_image_paths = all_image_paths[num_train_images:]
        return image_paths, validation_image_paths

    def resize(self, img, size):
        if img.shape[1] > size[0] or img.shape[0] > size[1]:
            return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    def predict(self, img, print_z=False):
        img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
        if self.denoise:
            img = self.train_data_generator.random_adjust(img)
        x = np.asarray(img).reshape((1,) + self.input_shape).astype('float32') / 255.0
        z = self.encoder.predict_on_batch(x=x).reshape(-1)
        if print_z:
            print(z)
        z = z.reshape((1,) + z.shape)
        y = self.decoder.predict_on_batch(x=z)
        y = np.asarray(y).reshape(self.input_shape) * 255.0
        decoded_img = np.clip(y, 0.0, 255.0).astype('uint8')
        return img, decoded_img

    def predict_images(self, image_paths):
        """
        Equal to the evaluate function. image paths are required.
        """
        if type(image_paths) is str:
            image_paths = glob(image_paths)
        image_paths = natsort.natsorted(image_paths)
        with tf.device('/cpu:0'):
            for path in image_paths:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.input_shape[-1] == 1 else cv2.IMREAD_COLOR)
                img, output_image = self.predict(img, print_z=True)
                img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
                img = np.asarray(img).reshape(img.shape[:2] + (self.input_shape[-1],))
                cv2.imshow('ae', np.concatenate((img, output_image), axis=1))
                key = cv2.waitKey(0)
                if key == 27:
                    break

    def predict_train_images(self):
        self.predict_images(self.train_image_paths)

    def predict_validation_images(self):
        self.predict_images(self.validation_image_paths)

    def training_view_function(self):
        """
        During training, the image is forwarded in real time, showing the results are shown.
        """
        cur_time = time()
        if cur_time - self.live_view_previous_time > 0.5:
            self.live_view_previous_time = cur_time
            if self.view_flag == 1:
                img_path = random.choice(self.train_image_paths)
                win_name = 'ae train data'
                self.view_flag = 0
            else:
                img_path = random.choice(self.validation_image_paths)
                win_name = 'ae validation data'
                self.view_flag = 1
            input_shape = self.ae.input_shape[1:]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR)
            img, output_image = self.predict(img)
            img = self.resize(img, (input_shape[1], input_shape[0]))
            cv2.imshow(win_name, np.concatenate((img.reshape(input_shape), output_image), axis=1))
            cv2.waitKey(1)
