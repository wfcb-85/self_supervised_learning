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

from alexnet_extraction import AlexNetMod

from matplotlib import pyplot as plt

from dataloader_functions import jigsaw_dataset

import resnet_extraction as resnet_extraction

from tqdm import tqdm

def compareModels(model1, model2):

    models_differ = 0

    list_of_mismatched_layers = []
    list_of_unchanged_layers = []

    for keyItem1, keyItem2 in zip(model1.state_dict().items(),
                                  model2.state_dict().items()):

        if torch.equal(keyItem1[1], keyItem2[1]):
            if (keyItem1[0] == keyItem2[0]):
                list_of_unchanged_layers.append(keyItem1[0])
        else:
            models_differ += 1
            if (keyItem1[0] == keyItem2[0]):
                # print("Mismatch found at ", keyItem1[0])
                list_of_mismatched_layers.append(keyItem1[0])
            else:
                raise Exception
    if models_differ == 0:
        print("Models Match perfectly")
    else:
        print("list of mismatched layers ", list_of_mismatched_layers)
        print("list of unchangeed layers ", list_of_unchanged_layers)
        print("-" * 20)

# data_dir = "/home/william/faces_poses/data/hymenoptera_data"
data_dir = "/home/william/Bureau/datasets/celeba_faces/train_test_split"

# Data augmentation and normalization for training
# Just normalization for validation

train_test = data_dir + "/train"

print("files in train ", len(os.listdir(train_test)))

print(os.listdir(train_test)[:50])

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x))
                  for x in ['train', 'test']}

print("DONT FORGET TO ADD THE TEST SET !!!")

# class_names = image_datasets['train'].classes

batch_size = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataSet = {x: jigsaw_dataset(image_datasets[x], batch_size=batch_size,
                                         shuffle=True, num_workers=1)
                  for x in ['train', 'test']}

# dataset_sizes = {x: dataloader[x].__len__() for x in ['train', 'test']}
dataset_sizes = {x: dataSet[x].__len__() for x in ['train', 'test']}

print("dataset sizes ", dataset_sizes)

# datasetTrain = dataloader["train"]
# datasetVal = dataloader['val']
#
#
dataloader_object = {'train': torch.utils.data.DataLoader(dataset=dataSet["train"],
                                         batch_size=batch_size,
                                         shuffle=True),

                     'test': torch.utils.data.DataLoader(dataset=dataSet["test"],
                                         batch_size=batch_size,
                                         shuffle=True)}
#
# dataloader_object = {'train': torch.utils.data.DataLoader(dataset=datasetTrain,
#                                          batch_size=batch_size,
#                                          shuffle=True,
#                                          collate_fn= _collate_fun)}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25,
                model_name = "Alexnet"):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # ERASE ME AFTERWARDS

    # temporal_model = models.resnet18(pretrained=False)
    # num_ftrs = temporal_model.fc.in_features
    # temporal_model.fc = nn.Linear(num_ftrs, 8)
    # temporal_model.to(device)
    # ERASE ME AFTERWARDS

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.

            counter_items = 0

            # temporal_model.load_state_dict(model.state_dict())

            for batch in dataloader_object[phase]:

                inputsCentralCrops = batch[0].to(device)
                inputsRandomCrops = batch[1].to(device)

                labels = batch[2].to(device)

                # print("labels shape : ", labels.shape)

                labels = labels.view(labels.shape[0])

                # print("labels shape after : ", labels.shape)

                # print("crops shapes ", inputsCentralCrops.shape,
                #       inputsRandomCrops.shape)

                # print("inputs size ", inputs.size(0))
                # print("inputs shape : ", inputs.shape)
                # print("outputs shape : ", labels.shape)
                # zero the parameter gradients

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # outputs = model(inputs)

                    if model_name == "Alexnet":

                        first_extract = model.forward_extraction(inputsCentralCrops)
                        second_extract = model.forward_extraction(inputsRandomCrops)

                        # print("extract shapes ", first_extract.shape, second_extract.shape)
                        fused_output = model.forward_fuse(first_extract, second_extract)

                    elif model_name == "resnet18":

                        first_extract = model(inputsCentralCrops)
                        second_extract = model(inputsRandomCrops)

                        fused_output = model.fused_output(first_extract, second_extract)

                    # print("fused output shape : ", fused_output.shape)
                    # print("fused output : ", fused_output)

                    # print("fused output shape : ", fused_output.shape)

                    _, preds = torch.max(fused_output, 1)

                    # print("model outputs shape: ", outputs.shape)

                    # print("outputs and labels shape {} {}".format(outputs.shape,
                    #                                               labels.shape))

                    # print("labels ", labels)
                    # print("preds : ", preds)

                    loss = criterion(fused_output, labels)

                    # print("loss ", loss)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # print("now we compare the previous model with the current one")
                    # compareModels(model, temporal_model)

                # statistics
                running_loss += loss.item() * inputsCentralCrops.size(0)
                counter_items += inputsCentralCrops.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (dataset_sizes[phase] )
            epoch_acc = running_corrects.double() / (dataset_sizes[phase])

            # print("dataset size of phase ", dataset_sizes[phase])
            # print("counter items ", counter_items)

            # print("dataset size of phase ", dataset_sizes[phase])
            # print("counter items ", counter_items)

            # print("running examples ", counter_items)
            # print("running loss ", running_loss)
            # print("running corrects ", running_corrects)
            # print("dataset size ", dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device : ", device)

alex_model = AlexNetMod()

# print("parameters : ", alex_model.state_dict())


# model_ft = models.resnet18(pretrained=True)

# num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 4)

alex_model = alex_model.to(device)

resnet18Model = resnet_extraction.resnet50(pretrained=True)

num_ftrs = resnet18Model.fc.in_features
resnet18Model.fc = nn.Linear(num_ftrs, 8)

resnet18Model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(resnet18Model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# model_ft = train_model(alex_model, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=1000, model_modified=False)

model_ft = train_model(resnet18Model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=1000, model_name="resnet18")