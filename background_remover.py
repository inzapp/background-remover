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
import cv2
import natsort
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from time import time
from model import Model
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ace import AdaptiveCrossentropy


class TrainingConfig:
    def __init__(self,
                 pretrained_model_path='',
                 train_image_path='',
                 validation_image_path='',
                 background_type='black',
                 model_name='model',
                 input_shape=(128, 128, 1),
                 lr=0.001,
                 warm_up=0.5,
                 batch_size=32,
                 iterations=100000,
                 denoise=False,
                 training_view=False):
        self.pretrained_model_path = pretrained_model_path
        self.train_image_path = train_image_path
        self.validation_image_path = validation_image_path
        self.background_type = background_type
        self.model_name = model_name
        self.input_shape = input_shape
        self.lr = lr
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.iterations = iterations
        self.denoise = denoise
        self.training_view = training_view


class BackgroundRemover:
    def __init__(self, config):
        self.config = config
        self.view_flag = 1
        self.pretrained_iteration_count = 0
        self.live_view_previous_time = time()
        self.checkpoint_path = 'checkpoint'

        use_input_layer_concat = self.config.background_type in ['black', 'gray', 'white', 'dark'] and not self.config.denoise
        self.model = Model(input_shape=self.config.input_shape, lr=self.config.lr, input_layer_concat=use_input_layer_concat)
        if self.config.pretrained_model_path != '':
            if os.path.exists(self.config.pretrained_model_path) and os.path.isfile(self.config.pretrained_model_path):
                print(f'\npretrained model path : [{self.config.pretrained_model_path}]')
                self.ae, self.config.input_shape = self.model.load(self.config.pretrained_model_path)
            else:
                print(f'\npretrained model not found : [{self.config.pretrained_model_path}]')
                exit(0)
        else:
            self.ae = self.model.build()

        self.train_image_paths = self.init_image_paths(self.config.train_image_path)
        self.validation_image_paths = self.init_image_paths(self.config.validation_image_path)

        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.train_data_generator = DataGenerator(
            image_paths=self.train_image_paths,
            input_shape=self.config.input_shape,
            batch_size=self.config.batch_size,
            add_noise=self.config.denoise,
            background_type=self.config.background_type)
        self.validation_data_generator = DataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=self.config.input_shape,
            batch_size=self.config.batch_size,
            add_noise=self.config.denoise,
            background_type=self.config.background_type)
        self.validation_data_generator_one_batch = DataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=self.config.input_shape,
            batch_size=1,
            add_noise=self.config.denoise,
            background_type=self.config.background_type)

    def load_model(self, model_path):
        if os.path.exists(model_path) and os.path.isfile(model_path):
            self.pretrained_iteration_count = self.parse_pretrained_iteration_count(model_path)
            self.ae, self.config.input_shape = self.model.load(model_path)
        else:
            print(f'pretrained model not found : {model_path}')
            exit(0)

    def parse_pretrained_iteration_count(self, pretrained_model_path):
        iteration_count = 0
        sp = f'{os.path.basename(pretrained_model_path)[:-3]}'.split('_')
        for i in range(len(sp)):
            if sp[i] == 'iter' and i > 0:
                try:
                    iteration_count = int(sp[i-1])
                except:
                    pass
                break
        return iteration_count

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    def check_forwarding_time(self):
        from time import perf_counter
        input_shape = self.ae.input_shape[1:]
        mul = 1
        for val in input_shape:
            mul *= val

        forward_count = 32
        noise = np.random.uniform(0.0, 1.0, mul * forward_count)
        noise = np.asarray(noise).reshape((forward_count, 1) + input_shape).astype('float32')
        with tf.device('/gpu:0'):
            self.graph_forward(self.ae, noise[0])  # only first forward is slow, skip first forward in check forwarding time

        print('\nstart test forward for check forwarding time.')
        with tf.device('/gpu:0'):
            st = perf_counter()
            for i in range(forward_count):
                self.graph_forward(self.ae, noise[i])
            et = perf_counter()
        forwarding_time = ((et - st) / forward_count) * 1000.0
        print(f'model forwarding time : {forwarding_time:.2f} ms')

    @tf.function
    def compute_gradient(self, model, optimizer, x, y, m):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            y = x * m
            abs_error = tf.abs(y - y_pred)
            mae = tf.reduce_mean(abs_error)
            ace = AdaptiveCrossentropy()(y, y_pred)
            pos_loss = ace * m
            neg_mask = tf.where(m == 0.0, 1.0, 0.0)
            ign_mask = tf.where(abs_error < 0.005, 0.0, 1.0) * neg_mask
            neg_loss = ace * ign_mask * tf.clip_by_value(abs_error, 0.25, 1.0)
            loss = tf.reduce_mean(pos_loss + neg_loss, axis=0)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mae

    def evaluate(self, generator):
        loss_sum = 0.0
        for batch_x, batch_y, batch_m in tqdm(generator):
            y = self.graph_forward(self.ae, batch_x)
            loss_sum += np.mean(np.abs(batch_y[0] - y[0]))
        return float(loss_sum / len(generator))

    def train(self):
        self.model.summary()
        self.check_forwarding_time()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print(f'validate on {len(self.validation_image_paths)} samples.')
        print('start training')
        iteration_count = self.pretrained_iteration_count
        optimizer = tf.keras.optimizers.RMSprop(lr=self.config.lr)
        lr_scheduler = LRScheduler(lr=self.config.lr, iterations=self.config.iterations, warm_up=self.config.warm_up, policy='step')
        while True:
            for batch_x, batch_y, batch_m in self.train_data_generator:
                lr_scheduler.update(optimizer, iteration_count)
                iteration_count += 1
                loss = self.compute_gradient(self.ae, optimizer, batch_x, batch_y, batch_m)
                print(f'\r[iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='\t')
                if self.config.training_view:
                    self.training_view_function()
                if iteration_count % 5000 == 0:
                    loss = self.evaluate(generator=self.validation_data_generator_one_batch)
                    self.model.save(self.checkpoint_path, self.config.model_name, iteration_count, loss, verbose=True)
                    print(f'[{iteration_count} iter] val_loss : {loss:.4f}\n')
                if iteration_count == self.config.iterations:
                    print('\n\ntrain end successfully')
                    return

    @staticmethod
    def init_image_paths(image_path):
        return glob(f'{image_path}/**/*.jpg', recursive=True)

    def resize(self, img, size):
        if img.shape[1] > size[0] or img.shape[0] > size[1]:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        else:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    def predict(self, img):
        input_shape = self.ae.input_shape[1:]
        input_height, input_width, input_channel = input_shape
        img = self.resize(img, (input_width, input_height))
        if self.config.denoise:
            img = self.train_data_generator.random_adjust(img)
        x = np.asarray(img).reshape((1,) + input_shape).astype('float32') / 255.0
        y = self.ae.predict_on_batch(x=x)
        y = np.asarray(y).reshape(input_shape) * 255.0
        decoded_img = np.clip(y, 0.0, 255.0).astype('uint8')
        return img, decoded_img

    def predict_images(self, dataset='validation', path='', width=0, height=0):
        input_height, input_width, input_channel = self.ae.input_shape[1:]
        if path != '':
            if not os.path.exists(path):
                print(f'path not exists : [{path}]')
                return
            if os.path.isfile(path):
                if path.endswith('.jpg'):
                    image_paths = [path]
                else:
                    print('invalid extension. jpg is available extension only')
                    return
            elif os.path.isdir(path):
                image_paths = glob(f'{path}/*.jpg')
            else:
                print(f'invalid file format : [{path}]')
                return
        else:
            assert dataset in ['train', 'validation']
            if dataset == 'train':
                image_paths = self.train_image_paths
            elif dataset == 'validation':
                image_paths = self.validation_image_paths
        if len(image_paths) == 0:
            print('no image found')
            return

        view_width, view_height = 0, 0
        if width > 0 and height > 0:
            view_width, view_height = width, height
        else:
            view_width, view_height = input_width, input_height
        view_size = (view_width, view_height)
        view_shape = (view_height, view_width, input_channel)
        for path in image_paths:
            print(f'image path : {path}')
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if input_channel == 1 else cv2.IMREAD_COLOR)
            img, output_image = self.predict(img)
            img = self.resize(img, view_size)
            output_image = self.resize(output_image, view_size)
            img = np.asarray(img).reshape(view_shape)
            output_image = np.asarray(output_image).reshape(view_shape)
            cv2.imshow('ae', np.concatenate((img, output_image), axis=1))
            key = cv2.waitKey(0)
            if key == 27:
                break

    def predict_video(self, path, width=0, height=0):
        if not path.startswith('rtsp://') and not (os.path.exists(path) and os.path.isfile(path)):
            print(f'video not found. video path : {path}')
            return
        cap = cv2.VideoCapture(path)
        input_height, input_width, input_channel = self.ae.input_shape[1:]
        view_width, view_height = 0, 0
        if width > 0 and height > 0:
            view_width, view_height = width, height
        else:
            view_width, view_height = input_width, input_height
        view_size = (view_width, view_height)
        view_shape = (view_height, view_width, input_channel)
        while True:
            frame_exist, bgr = cap.read()
            if not frame_exist:
                print('frame not exists')
                break
            img_color_converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY if input_channel == 1 else cv2.COLOR_BGR2RGB)
            img_x = self.resize(img_color_converted, (input_width, input_height))
            _, output_image = self.predict(img_x)
            img_view_input = np.asarray(self.resize(img_color_converted, view_size)).reshape(view_shape)
            img_view_output = np.asarray(self.resize(output_image, view_size)).reshape(view_shape)
            if input_channel == 3:
                img_view_input = cv2.cvtColor(img_view_input, cv2.COLOR_RGB2BGR)
                img_view_output = cv2.cvtColor(img_view_output, cv2.COLOR_RGB2BGR)
            cv2.imshow('video', np.concatenate((img_view_input, img_view_output), axis=1))
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def training_view_function(self):
        """
        During training, the image is forwarded in real time, showing the results are shown.
        """
        cur_time = time()
        if cur_time - self.live_view_previous_time > 0.5:
            self.live_view_previous_time = cur_time
            if self.view_flag == 1:
                img_path = np.random.choice(self.train_image_paths)
                win_name = 'ae train data'
                self.view_flag = 0
            else:
                img_path = np.random.choice(self.validation_image_paths)
                win_name = 'ae validation data'
                self.view_flag = 1
            input_shape = self.ae.input_shape[1:]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR)
            img, output_image = self.predict(img)
            img = self.resize(img, (input_shape[1], input_shape[0]))
            cv2.imshow(win_name, np.concatenate((img.reshape(input_shape), output_image), axis=1))
            cv2.waitKey(1)

