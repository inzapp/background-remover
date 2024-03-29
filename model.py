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
import os
import numpy as np
import tensorflow as tf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, lr, input_layer_concat):
        assert input_shape[-1] in [1, 3]
        self.input_shape = input_shape
        self.lr = lr
        self.ae = None
        self.input_layer_concat = input_layer_concat

    def load(self, model_path):
        self.ae = tf.keras.models.load_model(model_path, compile=False)
        self.input_shape = self.ae.input_shape[1:]
        return self.ae, self.input_shape

    def build(self):
        filters = 16
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.conv2d(input_layer, filters=filters * 1, kernel_size=3)
        f0 = x
        x = self.max_pool(x)

        x = self.conv2d(x, filters=filters * 2, kernel_size=3)
        f1 = x
        x = self.max_pool(x)
        
        x = self.conv2d(x, filters=filters * 4, kernel_size=3)
        x = self.conv2d(x, filters=filters * 4, kernel_size=3)
        f2 = x
        x = self.max_pool(x)

        x = self.conv2d(x, filters=filters * 8, kernel_size=3)
        x = self.conv2d(x, filters=filters * 8, kernel_size=3)
        f3 = x
        x = self.max_pool(x)

        x = self.conv2d(x, filters=filters * 16, kernel_size=3)
        x = self.conv2d(x, filters=filters * 16, kernel_size=3)
        f4 = x
        x = self.max_pool(x)

        x = self.conv2d(x, filters=filters * 32, kernel_size=3)
        x = self.conv2d(x, filters=filters * 32, kernel_size=3)
        x = self.conv2d(x, filters=filters * 16, kernel_size=1)
        x = self.upsampling(x)

        x = self.add([x, f4])
        x = self.conv2d(x, filters=filters * 16, kernel_size=3)
        x = self.conv2d(x, filters=filters * 16, kernel_size=3)
        x = self.conv2d(x, filters=filters * 8, kernel_size=1)
        x = self.upsampling(x)

        x = self.add([x, f3])
        x = self.conv2d(x, filters=filters * 8, kernel_size=3)
        x = self.conv2d(x, filters=filters * 8, kernel_size=3)
        x = self.conv2d(x, filters=filters * 4, kernel_size=1)
        x = self.upsampling(x)

        x = self.add([x, f2])
        x = self.conv2d(x, filters=filters * 4, kernel_size=3)
        x = self.conv2d(x, filters=filters * 4, kernel_size=3)
        x = self.conv2d(x, filters=filters * 2, kernel_size=1)
        x = self.upsampling(x)

        x = self.add([x, f1])
        x = self.conv2d(x, filters=filters * 2, kernel_size=3)
        x = self.conv2d(x, filters=filters * 1, kernel_size=1)
        x = self.upsampling(x)

        x = self.add([x, f0])
        x = self.conv2d(x, filters=filters * 1, kernel_size=3)
        if self.input_layer_concat:
            x = self.concat([x, input_layer])
        x = self.segmentation_layer(x)
        self.ae = tf.keras.models.Model(input_layer, x)
        self.ae.save('model.h5', include_optimizer=False)
        return self.ae

    def max_pool(self, x):
        return tf.keras.layers.MaxPool2D()(x)
    
    def upsampling(self, x):
        return tf.keras.layers.UpSampling2D()(x)

    def add(self, layers):
        return tf.keras.layers.Add()(layers)

    def concat(self, layers):
        return tf.keras.layers.Concatenate()(layers)

    def kernel_initializer(self):
        return tf.keras.initializers.glorot_normal()

    def conv2d(self, x, filters, kernel_size):
        return tf.keras.layers.Conv2D(
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            kernel_initializer=self.kernel_initializer(),
            activation='relu')(x)

    def segmentation_layer(self, x, name='output'):
        return tf.keras.layers.Conv2D(
            filters=self.input_shape[-1],
            padding='same',
            kernel_size=1,
            kernel_initializer=self.kernel_initializer(),
            activation='sigmoid')(x)

    def save(self, path, name, iteration_count, loss, verbose):
        save_path = f'{path}/{name}_{iteration_count}_iter_{loss:.4f}_loss.h5'
        self.ae.save(save_path, include_optimizer=False)
        if verbose:
            print(f'save success to {save_path}')

    def summary(self):
        self.ae.summary()

