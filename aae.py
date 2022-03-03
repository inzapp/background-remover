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


class AdversarialAutoEncoder:
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
                 pretrained_ae_path='',
                 pretrained_discriminator_path=''):
        self.iterations = iterations
        self.training_view = training_view
        self.live_view_previous_time = time()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        self.model = Model(input_shape=input_shape, lr=lr, momentum=momentum, encoding_dim=encoding_dim)
        if self.exists(pretrained_ae_path) and self.exists(pretrained_discriminator_path):
            print(f'\npretrained ae path : {[pretrained_ae_path]}')
            print(f'pretrained discriminator path : {[pretrained_discriminator_path]}')
            self.ae, self.discriminator, self.aae = self.model.load(pretrained_ae_path, pretrained_discriminator_path)
        else:
            self.ae, self.discriminator, self.aae = self.model.build()

        if validation_image_path != '':
            self.train_image_paths, _ = self.init_image_paths(train_image_path)
            self.validation_image_paths, _ = self.init_image_paths(validation_image_path)
        elif validation_split > 0.0:
            self.train_image_paths, self.validation_image_paths = self.init_image_paths(train_image_path, validation_split=validation_split)

        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.train_data_generator = AAEDataGenerator(
            image_paths=self.train_image_paths,
            input_shape=input_shape,
            batch_size=batch_size)
        self.validation_data_generator = AAEDataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=input_shape,
            batch_size=batch_size)

    def exists(self, path):
        return os.path.exists(path) and os.path.isfile(path)

    def fit(self):
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print(f'validate on {len(self.validation_image_paths)} samples.')
        print('start training')
        self.train()

    def train(self):
        iteration_count = 0
        half_batch_size = self.batch_size // 2
        while True:
            self.train_data_generator.shuffle()
            for ae_x, ae_y in self.train_data_generator:
                iteration_count += 1
                self.ae.trainable = True
                ae_loss = self.ae.train_on_batch(ae_x, ae_y, return_dict=True)['loss']
                half_ae_x = ae_x[:half_batch_size]
                half_ae_y = self.ae.predict_on_batch(half_ae_x)
                discriminator_x = np.append(half_ae_x, half_ae_y, axis=0)
                discriminator_y = np.append(np.ones(shape=(half_batch_size, 1)), np.zeros(shape=(half_batch_size, 1)), axis=0).astype('float32')
                r = np.arange(self.batch_size)
                np.random.shuffle(r)
                discriminator_x = discriminator_x[r]
                discriminator_y = discriminator_y[r]
                discriminator_loss = self.discriminator.train_on_batch(discriminator_x, discriminator_y, return_dict=True)['loss']
                aae_x = discriminator_x
                aae_y = np.ones(shape=(self.batch_size, 1), dtype=np.float32)
                self.ae.trainable = False
                aae_loss = self.aae.train_on_batch(aae_x, aae_y, return_dict=True)['loss']
                print(f'\r[iteration count : {iteration_count:6d}] ae loss => {ae_loss:.4f}, discriminator loss => {discriminator_loss:.4f}, aae loss => {aae_loss:.4f}', end='\t')
                if self.training_view:
                    self.training_view_function()
                if iteration_count % 1000 == 0:
                    self.model.save(self.checkpoint_path, iteration_count, ae_loss, discriminator_loss)
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

    def predict(self, img):
        img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
        x = np.asarray(img).reshape((1,) + self.ae.input_shape[1:]).astype('float32') / 255.0
        y = self.ae.predict_on_batch(x=x)
        y = np.asarray(y).reshape(self.ae.input_shape[1:]) * 255.0
        output_img = np.clip(y, 0.0, 255.0).astype('uint8')
        return output_img

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
                output_image = self.predict(img)
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
            if np.random.uniform() > 0.5:
                img_path = random.choice(self.train_image_paths)
            else:
                img_path = random.choice(self.validation_image_paths)
            input_shape = self.ae.input_shape[1:]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR)
            output_image = self.predict(img)
            img = self.resize(img, (input_shape[1], input_shape[0]))
            cv2.imshow('ae', np.concatenate((img.reshape(input_shape), output_image), axis=1))
            cv2.waitKey(1)
