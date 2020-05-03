from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets
import time
import os
import copy

from z_module import z_module

from dataloader_functions import simclrDataset

import resnet_extraction as resnet_extraction

from lossFunction import simCLRLossFunction

# References :
# https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603


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

data_dir = "/home/william/Bureau/datasets/clba/train_test_split"

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

dataSet = {x: simclrDataset(image_datasets[x], batch_size=batch_size,
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



def train_model(featureModel, zModel, criterion, optimizer, scheduler, num_epochs=25,
                model_name = "Alexnet"):

    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e6

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:

            if phase == 'train':
                featureModel.train()  # Set model to training mode
                zModel.train()
            else:
                featureModel.eval()   # Set model to evaluate mode
                zModel.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.

            counter_items = 0

            # resnetTemporal.load_state_dict(featureModel.state_dict())
            # zTemporal.load_state_dict(zModel.state_dict())

            for batch in dataloader_object[phase]:

                firstAugmentation = batch[0].to(device)
                secondAugmentation = batch[1].to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):


                    first_extract = featureModel(firstAugmentation)
                    second_extract = featureModel(secondAugmentation)


                    zFirstOutput = zModel(first_extract)
                    zSecondOutput = zModel(second_extract)


                    loss = simCLRLossFunction(zFirstOutput, zSecondOutput, criterion)

                    print("loss ", loss)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * firstAugmentation.shape[0]
                counter_items += firstAugmentation.shape[0]
                # running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (dataset_sizes[phase] )

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_featureModel_wts = copy.deepcopy(featureModel.state_dict())
                best_zedModel_wts = copy.deepcopy(z_model.state_dict())

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

# resnet18Model = torchvision.models.resnet18(pretrained=True)

# These three should be parameters

resnetArchitecture = 50
zOutputSize= 4096
HSize = 2048

if resnetArchitecture == 50:

    zInputSize = 2048

resnetModel = resnet_extraction.resnet50(pretrained=True)

z_model = z_module(D_in=zInputSize, H = HSize, D_out=zOutputSize)

resnetModel.to(device)
z_model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(list(resnetModel.parameters()) + list(z_model.parameters()),
                         lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# model_ft = train_model(alex_model, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=1000, model_modified=False)

model_ft = train_model(resnetModel, z_model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=1000, model_name="resnet18")