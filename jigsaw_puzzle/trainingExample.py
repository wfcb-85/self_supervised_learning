from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets
import os
from alexnet_extraction import AlexNetMod
from dataloader_functions import jigsaw_dataset
import resnet_extraction as resnet_extraction

from trainingLoop import train_model

data_dir = "/home/william/Bureau/datasets/clba/train_test_split"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x))
                  for x in ['train', 'test']}

batch_size = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataSet = {x: jigsaw_dataset(image_datasets[x], batch_size=batch_size,
                                         shuffle=True, num_workers=1)
                  for x in ['train', 'test']}

dataset_sizes = {x: dataSet[x].__len__() for x in ['train', 'test']}

dataloader_object = {'train': torch.utils.data.DataLoader(dataset=dataSet["train"],
                                         batch_size=batch_size,
                                         shuffle=True),

                     'test': torch.utils.data.DataLoader(dataset=dataSet["test"],
                                         batch_size=batch_size,
                                         shuffle=True)}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device : ", device)

resnet18Model = resnet_extraction.resnet18(pretrained=True)

num_ftrs = resnet18Model.fc.in_features
resnet18Model.fc = nn.Linear(num_ftrs, 8)

resnet18Model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(resnet18Model.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(resnet18Model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=1000, model_name="resnet18")