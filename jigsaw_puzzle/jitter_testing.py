#https://stackoverflow.com/questions/35152636/random-flipping-and-rgb-jittering-slight-value-change-of-image

import cv2
import numpy as np
from pylab import *

import matplotlib.pyplot as plt

import random

import torch

img = cv2.imread('/home/william/masa.jpg' )
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 defaul color code is BGR
h,w,c = img.shape # (768, 1024, 3)

noise = np.random.randint(0,50,(h, w)) # design jitter/noise here
zitter = np.zeros_like(img)
zitter[:,:,1] = noise

# noise_added = cv2.add(img, zitter)
# combined = np.vstack((img[:int(h/2),:,:], noise_added[int(h/2):,:,:]))

combined = torch.Tensor(img)
# combined.__add__((torch.rand_like(combined)-0.5) * 0.0)

combined = np.array(combined)
# combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

# plt.imshow(combined, interpolation='none')
# plt.show()

a = torch.rand(2,2)

print(a)

print(torch.rand_like(a))

"""
JITTER ALGORITHM:

Once we know the size of the image and the magnitude of the jitter, we can proceed
to stablish the number of jitters. For instance, let us say our tensor is one-dimensional

"""
tensor_size = 15
jitter_size = 4

number_of_zones = int(tensor_size/jitter_size)

zones = []
# zones.append([0,jitter_size])

for i in range(number_of_zones):
    # zones.append([i*(jitter_size +1) -1, (i) + ((i+1) * jitter_size) -1])
    zones.append([i*jitter_size, (i+1) * jitter_size])

# zones.append([tensor_size -jitter_size +1, tensor_size -1])

print("zones : ", zones)

rand_tens = torch.rand(15,1)

permutation = torch.randperm(jitter_size)
print("permutation :", permutation)

# We will have the same random perm for all subzones, because IMO
# the fact that each subzone has a different permutation does not
# add that much to the training

print(rand_tens)
temp_tensor= rand_tens.clone()

temp_tensor = temp_tensor.view(-1)

for zone in zones:
    permutation = torch.randperm(jitter_size)
    print("zone : ", zone)
    loc_permutation = permutation + zone[0]
    print("permutation ", loc_permutation)
    print(rand_tens.view(-1)[loc_permutation])

    # temp_tensor.append(rand_tens.view(-1)[loc_permutation])
    print(zone[0], zone[1])
    # print(rand_tens.view(-1)[loc_permutation].view())
    temp_tensor[zone[0] : zone[1]] = rand_tens.view(-1)[loc_permutation]

# temp_tensor[zones[-1][0] : zones[-1][1]] = rand_tens[zones[-1][0] : zones[-1][1]]

# print(rand_tens)
print("random tensor : ",rand_tens)
print("temporal tensor : ", temp_tensor)

# print(rand_tens[0:3])
# print(rand_tens[3:6])
# print(rand_tens[6:9])

from torch import randperm
# import torch
#
# a = torch.rand(10, 5)
#
# print(a)
#
# print(a.size())
#
# a = a[torch.randperm(3)]
# print(a)
#
# size_of_image = 80
# size_of_jitter = 2
#
# number_of_sub_images = size_of_image / size_of_jitter
#
# print(torch.rand(size_of_jitter) )