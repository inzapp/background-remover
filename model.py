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
import os
import numpy as np
import tensorflow as tf
from keras import backend as K


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, lr, momentum, encoding_dim):
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.encoding_dim = encoding_dim
        self.ae = None

    def load(self, model_path):
        self.ae = tf.keras.models.load_model(model_path, compile=False)
        self.ae.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.momentum), loss=self.loss)
        self.input_shape = self.ae.input_shape[1:]
        self.encoding_dim = 0
        for layer in self.ae.layers:
            if layer.name == 'encoder_output':
                self.encoding_dim = layer.units
                break
        if self.encoding_dim == 0:
            exit(0)
        return self.ae, self.input_shape, self.encoding_dim

    def build(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=5, kernel_initializer='he_normal', activation='relu')(input_layer)
        x = tf.keras.layers.SpatialDropout2D(0.125)(x)
        x = tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=5, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.SpatialDropout2D(0.125)(x)
        x = tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=1, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.SpatialDropout2D(0.125)(x)
        x = tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=5, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.SpatialDropout2D(0.125)(x)
        x = tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=5, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.SpatialDropout2D(0.125)(x)
        x = tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=1, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.SpatialDropout2D(0.125)(x)
        x = tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=5, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.SpatialDropout2D(0.125)(x)
        x = tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=5, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.SpatialDropout2D(0.125)(x)
        x = tf.keras.layers.Conv2D(filters=self.input_shape[-1], padding='same', kernel_size=1, kernel_initializer='glorot_uniform', activation='sigmoid')(x)
        self.ae = tf.keras.models.Model(input_layer, x)
        self.ae.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.momentum), loss=self.loss)
        self.ae.save('model.h5', include_optimizer=False)
        return self.ae

    def loss(self, y_true, y_pred):
        return -K.log((1.0 + K.epsilon()) - K.abs(y_true - y_pred))

    def save(self, path, iteration_count, loss):
        self.ae.save(f'{path}/ae_{iteration_count}_iter_{loss:.4f}_loss.h5', include_optimizer=False)

    def summary(self):
        self.ae.summary()
