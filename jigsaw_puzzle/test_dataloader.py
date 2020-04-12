from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random

from dataloader_functions import jigsaw_dataset

print("IN THE JIGSAW_DATASET, EXCHANGE THE __getitem__ function for the "
      "__getitem_for_debugging__ function. Otherwise this script wont work.")

data_dir = "/home/william/Bureau/datasets/celeba_faces/train_test_split"

image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x))
                  for x in ['train', 'test']}

batch_size = 4

dataset = {x : jigsaw_dataset(image_datasets[x], shuffle=True)
              for x in ["train", "test"]}

print("dataset : ", dataset)

dataloader = torch.utils.data.DataLoader(dataset = dataset["train"])

print("dataloader : ", dataloader)

random_number = random.random() * 1000
count = 0

for b in dataloader:
    central, random, img = b
    count +=1

    if count > random_number:
        break


transf = transforms.ToPILImage()

plt.subplot(1,3,1)

print("image shape ", central.shape)
fig=plt.imshow(transf(central[0]))
fig.axes.get_xaxis().set_visible(False)

fig.axes.get_yaxis().set_visible(False)

plt.subplot(1,3,2)

print("image shape ", random.shape)
fig=plt.imshow(transf(random[0]))
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)


plt.subplot(1,3,3)

print("image shape ", img.shape)
fig=plt.imshow(transf(img[0]))
fig.axes.get_xaxis().set_visible(False)

fig.axes.get_yaxis().set_visible(False)

plt.show()