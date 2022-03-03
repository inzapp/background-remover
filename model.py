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
        self.ae, self.discriminator, self.aae = None, None, None

    def load(self, ae_path, discriminator_path):
        self.ae_body = tf.keras.models.load_model(ae_path, compile=False)
        self.ae = tf.keras.models.Sequential([
            self.ae_body,
            self.transformer()])
        self.ae.compile(optimizer=tf.keras.optimizers.Adam(lr=lr, beta_1=momentum), loss=self.loss)
        self.discriminator = tf.keras.models.load_model(discriminator_path, compile=False)
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(lr=lr, beta_1=momentum), loss=self.loss)
        self.ae.trainable = False
        self.aae = tf.keras.models.Sequential([
            self.ae,
            self.discriminator])
        self.aae.compile(optimizer=tf.keras.optimizers.Adam(lr=lr, beta_1=momentum), loss=self.loss)
        return self.ae, self.discriminator, self.aae

    def build(self):
        self.ae_body = tf.keras.models.Sequential([
            self.encoder(),
            self.gap(self.encoding_dim, 'relu'),
            self.decoder()])

        self.ae = tf.keras.models.Sequential([
            self.ae_body,
            self.transformer()])
        self.ae.build((None,) + self.input_shape)
        self.ae.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.momentum), loss=self.loss)

        self.discriminator = tf.keras.models.Sequential([
            self.encoder(),
            self.gap(1, 'sigmoid')])
        self.discriminator.build((None,) + self.input_shape)
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.momentum), loss=self.loss)

        self.discriminator.trainable = False
        self.aae = tf.keras.models.Sequential([
            self.ae,
            self.discriminator])
        self.aae.build((None,) + self.input_shape)
        self.aae.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.momentum), loss=self.loss)
        self.discriminator.trainable = True
        return self.ae, self.discriminator, self.aae

    def save(self, path, iteration_count, ae_loss, discriminator_loss):
        self.ae_body.save(f'{path}/ae_{iteration_count}_iter_{ae_loss:.4f}_loss.h5', include_optimizer=False)
        self.discriminator.save(f'{path}/discriminator_{iteration_count}_iter_{discriminator_loss:.4f}_loss.h5', include_optimizer=False)

    def summary(self):
        print('\nautoencoder summary')
        self.ae.summary()
        print('\ndiscriminator summary')
        self.discriminator.summary()
        print('\nadversarial autoencoder summary')
        self.aae.summary()

    def loss(self, y_true, y_pred):
        return -K.log((1.0 + K.epsilon()) - K.abs(y_true - y_pred))

    def transformer(self):
        return tf.keras.layers.Reshape(input_shape=(int(np.prod(self.input_shape)),), target_shape=self.input_shape)

    def encoder(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=16, padding='same', kernel_size=3, kernel_initializer='he_uniform', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.SpatialDropout2D(0.0625),
            tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=3, kernel_initializer='he_uniform', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.SpatialDropout2D(0.0625),
            tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=3, kernel_initializer='he_uniform', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.SpatialDropout2D(0.0625),
            tf.keras.layers.Conv2D(filters=128, padding='same', kernel_size=3, kernel_initializer='he_uniform', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.SpatialDropout2D(0.0625),
            tf.keras.layers.Conv2D(filters=256, padding='same', kernel_size=3, kernel_initializer='he_uniform', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SpatialDropout2D(0.0625),
            tf.keras.layers.Conv2D(filters=512, padding='same', kernel_size=3, kernel_initializer='he_uniform', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()])

    def decoder(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(units=512, kernel_initializer='he_uniform', activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(units=1024, kernel_initializer='he_uniform', activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(units=int(np.prod(self.input_shape)), kernel_initializer='glorot_uniform', activation='sigmoid', name='decoder_output')])

    def gap(self, filters, activation):
        if activation == 'relu':
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=1, kernel_initializer='he_uniform', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.GlobalAveragePooling2D(name='encoder_output')])
        elif activation == 'sigmoid':
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=filters, padding='same', kernel_size=1, kernel_initializer='glorot_uniform', activation='sigmoid'),
                tf.keras.layers.GlobalAveragePooling2D(name='discriminator_output')])
        else:
            return None
