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
import argparse

from background_remover import BackgroundRemover


if __name__ == '__main__':
    background_remover = BackgroundRemover(
        train_image_path=r'/train_data/coco/train',
        validation_image_path=r'/train_data/coco/validation',
        background_type='black',
        input_shape=(128, 128, 1),
        lr=0.001,
        warm_up=0.5,
        batch_size=4,
        iterations=200000,
        denoise=False,
        training_view=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true', help='prediction flag with pretrained model')
    parser.add_argument('--model', type=str, default='', help='pretrained model path for prediction')
    parser.add_argument('--path', type=str, default='', help='image or video path for prediction')
    parser.add_argument('--width', type=int, default=0, help='width for showing predicted result')
    parser.add_argument('--height', type=int, default=0, help='height for showing predicted result')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset name for prediction. train or validation')
    args = parser.parse_args()
    if args.model != '':
        background_remover.pretrained_model_path = args.model
    if args.predict:
        if args.path.endswith('.mp4') or args.path.startswith('rtsp://'):
            background_remover.predict_video(path=args.path, width=args.width, height=args.height)
        else:
            background_remover.predict_images(dataset=args.dataset, path=args.path, width=args.width, height=args.height)
    else:
        background_remover.train()

