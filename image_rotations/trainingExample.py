from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models
import os
from trainingLoop import train_model

from dataloader_functions import DataLoader, wc_rotation_DataSet, \
    _collate_fun, collate_fn_alter

data_dir = "/home/william/Bureau/datasets/clba/train_test_split"


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x))
                  for x in ['train', 'test']}

batch_size = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rotations_number = 4

dataloader = {x: wc_rotation_DataSet(image_datasets[x], batch_size=batch_size,
                                         shuffle=True, num_workers=1)
                  for x in ['train', 'test']}

dataset_sizes = {x: dataloader[x].__len__() for x in ['train', 'test']}


print("dataset sizes ", dataset_sizes)

dataloader_object = {'train': torch.utils.data.DataLoader(dataset=dataloader["train"],
                                         batch_size=batch_size,
                                         shuffle=True,
                                         collate_fn= _collate_fun),

                     'test': torch.utils.data.DataLoader(dataset=dataloader["test"],
                                         batch_size=batch_size,
                                         shuffle=True,
                                         collate_fn= _collate_fun)}



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device : ", device)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, rotations_number, criterion,dataloader,
                       dataset_sizes, optimizer_ft, exp_lr_scheduler, device,
                       num_epochs=25)
