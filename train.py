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
from ae import AutoEncoder

if __name__ == '__main__':
    AutoEncoder(
        train_image_path=r'/train_data/coco/train',
        validation_image_path=r'/train_data/coco/validation',
        input_shape=(128, 128, 1),
        lr=0.001,
        warm_up=0.5,
        batch_size=32,
        iterations=200000,
        training_view=True,
        remove_background=False,
        remove_background_type='black',
        vertical_shake_power=0,
        horizontal_shake_power=0,
        denoise=False).fit()
