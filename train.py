import argparse
import os
import time
from collections import deque

import torch
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model

import torch.optim as optim

from enum import Enum
from typing import Tuple

import PIL
import torch.utils.data
from PIL import Image
from torch import Tensor
from torchvision import datasets
from torchvision.transforms import transforms


def _train(path_to_data_dir: str, path_to_checkpoints_dir: str):
    os.makedirs(path_to_checkpoints_dir, exist_ok=True)

    learning_rate = 0.01
    # TODO: CODE BEGIN
    #raise NotImplementedError
    # dataset = XXX
    #dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TRAIN)
    data_transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root='data/processed',transform=data_transform)
    
    # dataloader = XXX
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f'label:{dataset.class_to_idx}')
    # TODO: CODE END

    # TODO: CODE BEGIN
    #raise NotImplementedError
    # model = XXX
    model = Model().cuda()
    #model = Model()
    # optimizer = XXX
    optimizer = optim.SGD(model.parameters(),lr = learning_rate, momentum = 0.9, weight_decay = 5e-4)
    # TODO: CODE END

    num_steps_to_display = 20
    num_steps_to_snapshot = 1000
    num_steps_to_finish = 10000

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False

    print('Start training')

    while not should_stop:
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.cuda()
            labels = labels.cuda()
            
            # TODO: CODE BEGIN
            #raise NotImplementedError
            # logits = XXX
            logits = model(images)
            # loss = XXX
            loss = model.loss(logits, labels)
            # TODO: CODE END

            # TODO: CODE BEGIN
            #raise NotImplementedError
            # optimizer.XXX
            optimizer.zero_grad()
            # loss.XXX
            loss.backward()
            # optimizer.XXX
            optimizer.step()
            # TODO: CODE END

            losses.append(loss.item())
            step += 1

            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                avg_loss = sum(losses) / len(losses)
                print(f'[Step {step}] Avg. Loss = {avg_loss:.6f} ({steps_per_sec:.2f} steps/sec)')

            if step % num_steps_to_snapshot == 0:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                print(f'Model saved to {path_to_checkpoint}')
            
            if step % 5000 == 0 :
                optimizer = optim.SGD(model.parameters(),lr = learning_rate*0.1)
                learning_rate = learning_rate*0.1
                print(f'Learning rate changed to {learning_rate}')

            if step == num_steps_to_finish:
                should_stop = True
                break

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
        args = parser.parse_args()

        path_to_data_dir = args.data_dir
        path_to_checkpoints_dir = args.checkpoints_dir

        _train(path_to_data_dir, path_to_checkpoints_dir)

    main()
