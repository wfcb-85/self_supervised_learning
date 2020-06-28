from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets
import os
from dataloader_functions import mocoDataset

from trainingLoop import train_model

import sys
# Adding parent folder
import resnet_extraction

import pdb

# References :
# https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603

data_dir = "/home/william/Bureau/datasets/clba/train_test_split"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x))
                  for x in ['train', 'test']}

batch_size = 4
dictionaryFactor = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataSet = {x: mocoDataset(image_datasets[x], batch_size=batch_size,
                                         shuffle=True, num_workers=1)
                  for x in ['train', 'test']}

# dataset_sizes = {x: dataloader[x].__len__() for x in ['train', 'test']}
dataset_sizes = {x: dataSet[x].__len__() for x in ['train', 'test']}

print("dataset sizes ", dataset_sizes)

dataloader_object = {'train': torch.utils.data.DataLoader(dataset=dataSet["train"],
                                         batch_size=batch_size,
                                         shuffle=True),

                     'test': torch.utils.data.DataLoader(dataset=dataSet["test"],
                                         batch_size=batch_size,
                                         shuffle=True)}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device : ", device)

resnetArchitecture = 50
zOutputSize= 4096

if resnetArchitecture == 18:
    resnetModelQuery = resnet_extraction.resnet18(pretrained=True)
    resnetModelKey = resnet_extraction.resnet18(pretrained=True)
    output_size = 512

elif resnetArchitecture == 34:
    resnetModelQuery = resnet_extraction.resnet34(pretrained=True)
    resnetModelKey = resnet_extraction.resnet34(pretrained=True)

elif resnetArchitecture == 50:
    resnetModelQuery = resnet_extraction.resnet50(pretrained=True)
    resnetModelKey = resnet_extraction.resnet50(pretrained=True)
    output_size = 2048

elif resnetArchitecture == 101:
    resnetModelQuery = resnet_extraction.resnet101(pretrained=True)
    resnetModelKey = resnet_extraction.resnet101(pretrained=True)

elif resnetArchitecture == 152:
    resnetModelQuery = resnet_extraction.resnet101(pretrained=True)
    resnetModelKey = resnet_extraction.resnet101(pretrained=True)


resnetModelQuery.to(device)
resnetModelKey.to(device)

criterion = nn.CrossEntropyLoss()


dictionarySize = batch_size * dictionaryFactor


optimizer_ft = optim.SGD(list(resnetModelQuery.parameters()),
                         lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(resnetModelQuery, resnetModelKey, dictionarySize, output_size, batch_size,
 dataloader_object, dataset_sizes,
                       criterion,
                       optimizer_ft, exp_lr_scheduler,
                       device,num_epochs=1000,  model_name="myMpdem")