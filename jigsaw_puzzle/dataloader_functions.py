import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import torchvision.transforms.functional as TF

import random

import numpy as np

import cv2

from PIL import Image

"""
https://arxiv.org/pdf/1505.05192.pdf

We sample random pairs of patches in one of eight spatial
configurations, and present each pair to a machine learner,
providing no information about the patches' original 
position within the image.

The algorithm must then guess the position of one patch
relative to the other.

Building a ConvNet that can predict a relative offset for
a pair of patches is, in principle, straightforward: the
network must feed the two input patches through several 
convolution layers, and produce an output that assigns 
a probability to each of the eight spatial configurations 
that might have been samples (i.e. a softmax output). Note,
however, that we ultimately wish to learn a feature embedding
for individual patches, such that patches which are similar
would be close in the embedding space.

To achieve this, we use a pair of AlexNet-style architectures
that process each patch separately, until a depth analogous
to fc6 in AlexNet, after which point the representations are
fused. For the layers that process only one of the patches,
weights are tied between both sides of the network,
such that the same fc6-level embedding function is computed
for both patches. 

To obtain training examples given an image, we sample the first 
patch uniformly, without any reference to image content. Given
the position of the first patch, we sample the second patch
randomly from the eight possible neighboring locations.

We also randomly jitter each patch location by up to 7 pixels.

For some images, another trivial solution exists. We traced the
problem to an unexpected culprit: chromatic aberration. Chromatic
aberration arises from differences in the way the lens focuses
light at different wavelengths. 

To deal with this problem, we experimented with two types of 
pre-processing. One is to shift green and magenta toward gray
('projection'). Specifically, let a = [-1, 2, -1] (the 'green-magenta
color axis' in RGB space). We then define B = I - a.T * a / (a * a.T),
which is a matrix that substracts the projection of a color onto the
green-magenta color axis. We multiply every pixel color by B. An 
alternative approach is to randomly drop 2 of the 3 color channels from
each patch ('color dropping'), replacing the dropped colors with 
Gaussian noise (standard deviation ~ 1/100 the standard deviation
of the remaining channel). 

IMPLEMENTATION DETAILS:

First,  we resize each image to between 150K and 450K total pixels,
preserving the aspect-ration. From these images, we sample patches at
resolution96-b-96. We allow a gap of 48 pixels between the sampled
patches in the grid, but also jitter the location of each patch in
the grid by -7 to 7 pixels in each direction. We preprocess patches by
(1) mean substraction (2) projecting or dropping colors, and (3) 
randomly downsampling some patches to as little as 100 total pixels, and
then upsampling it, to build robustness to pixelation.

Our final implementation employs batch normalization without the 
scale and shift (gamma and beta), which forces the network activations
to vary across examples.

"""

"""
JITTER ALGORITHM:

Once we know the size of the image and the magnitude of the jitter, we can proceed
to stablish the number of jitters. For instance, let us say our tensor is one-dimensional

tensor_size = n
jitter_size = s

number_of_zones = n/s

zones = []

for i in range(number_of_zones) -1:
    zones.append()
    



"""


class jitter(object):

    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, img):
        # Copied from https://github.com/fastai/fastai/blob/master/fastai/vision/transform.py
        return img.flow.add_((torch.rand_like(img.flow)-0.5)*self.magnitude*2)

def estimate_crop_position(p_crop_number,p_initialJigsawXPosition, p_initialJigsawYPosition,
                           x_c, y_c, d_c):
    """
    ----- ----- -----
    | 0 | | 1 | | 2 |
    ----- ----- -----
    | 3 | | 4 | | 5 |
    ----- ----- -----
    | 6 | | 7 | | 8 |
    ----- ----- -----

    Now we decide the coordinates of each of the crops. The coordinates for each crop will be
    as follows [initialPosX, finalPosX, initialPosY, finalPosY]

    """

    # print("x_c : {} , y_c : {} , d_c : {}".format(x_c, y_c, d_c))

    crop_positions = []

    cr_pos_0 =  [p_initialJigsawXPosition, p_initialJigsawXPosition + x_c,
                     p_initialJigsawYPosition, p_initialJigsawYPosition + y_c]

    cr_pos_1 = [p_initialJigsawXPosition + d_c +x_c, p_initialJigsawXPosition + (2 *x_c)  + d_c,
                     p_initialJigsawYPosition, p_initialJigsawYPosition + y_c]

    cr_pos_2 = [p_initialJigsawXPosition + (2 *x_c)  +  (2 * d_c),
                                 p_initialJigsawXPosition + (3 * x_c) + (2 * d_c),
                     p_initialJigsawYPosition, p_initialJigsawYPosition + y_c]

    cr_pos_3 = [p_initialJigsawXPosition, p_initialJigsawXPosition + x_c,
                     p_initialJigsawYPosition + y_c +  d_c,
                                 p_initialJigsawYPosition + (2 *y_c)  + d_c]

    cr_pos_4 = [p_initialJigsawXPosition + d_c +x_c,
                                 p_initialJigsawXPosition + (2 *x_c)  + d_c,
                                 p_initialJigsawYPosition + y_c + d_c,
                                 p_initialJigsawYPosition + (2 * y_c) + d_c]

    cr_pos_5 = [p_initialJigsawXPosition + (2 *x_c)  +  (2 * d_c),
                                 p_initialJigsawXPosition + (3 * x_c) + (2 * d_c),
                                 p_initialJigsawYPosition + y_c + d_c,
                                 p_initialJigsawYPosition + (2 * y_c) + d_c]

    cr_pos_6 = [p_initialJigsawXPosition, p_initialJigsawXPosition + x_c,
                                 p_initialJigsawYPosition + (2 * y_c) + (2 * d_c),
                                 p_initialJigsawYPosition + (3 * y_c) + (2 * d_c)]

    cr_pos_7 = [p_initialJigsawXPosition + d_c +x_c, p_initialJigsawXPosition + (2 *x_c)  + d_c,
                                 p_initialJigsawYPosition + (2 * y_c) + (2 * d_c),
                                 p_initialJigsawYPosition + (3 * y_c) + (2 * d_c)]

    cr_pos_8 = [p_initialJigsawXPosition + (2 *x_c)  +  (2 * d_c),
                                 p_initialJigsawXPosition + (3 * x_c) + (2 * d_c),
                                 p_initialJigsawYPosition + (2 * y_c) + (2 * d_c),
                                 p_initialJigsawYPosition + (3 * y_c) + (2 * d_c)]

    """
    print("crop position 1 : ", cr_pos_1 )
    print("crop position 2 : ", cr_pos_2)
    print("crop position 3 : ", cr_pos_3)
    print("crop position 4 : ", cr_pos_4)
    print("crop position 5 : ", cr_pos_5)
    print("crop position 6 : ", cr_pos_6)
    print("crop position 7 : ", cr_pos_7)
    print("crop position 8 : ", cr_pos_8)
    print("crop position 9 : ", cr_pos_9)
    """

    if p_crop_number == 0:
        crop_positions = cr_pos_0

    elif p_crop_number == 1:
        crop_positions = cr_pos_1

    elif p_crop_number == 2:
        crop_positions = cr_pos_2

    elif p_crop_number == 3:
        crop_positions = cr_pos_3

    elif p_crop_number == 4:
        crop_positions = cr_pos_4

    elif p_crop_number == 5:
        crop_positions = cr_pos_5

    elif p_crop_number == 6:
        crop_positions = cr_pos_6

    elif p_crop_number == 7:
        crop_positions = cr_pos_7

    elif p_crop_number == 8:
        crop_positions = cr_pos_8
    else:

        print("random crop position ")
        raise ValueError("Dont know what to do with this crop position : {}."
                         " The crop position must be an integer in [1-9]".format(p_crop_number))

    return crop_positions


def determine_upper_left_jigsaw_position(p_image_shape,
                                         p_jigsaw_shape):

    x_j, y_j = p_jigsaw_shape
    x, y = p_image_shape

    initialJigsawXPosition = int(random.random() * (x - x_j))
    initialJigsawYPosition = int(random.random() * (y - y_j))

    return initialJigsawXPosition, initialJigsawYPosition

def determine_jigsaw_shape(p_crop_shape, p_crops_distance):

    x_c, y_c = p_crop_shape

    return (x_c * 3 ) + (p_crops_distance * 2), \
           (y_c * 3) + (p_crops_distance *2)

def determine_the_two_crops_positions(p_image_shape, p_crop_shape,
                                     p_crops_distance):

    d_c = p_crops_distance
    x,y = p_image_shape
    x_c, y_c = p_crop_shape[0], p_crop_shape[1]

    # print("xc, y_c, d_c ", x_c, y_c, d_c)

    x_j, y_j = determine_jigsaw_shape(p_crop_shape, p_crops_distance)

    # Just checking the jigsaw matrix is not bigger than the image
    # itself.
    if x_j > x or y_j > y:

        print("image size : ({},{}) ".format(x, y))
        print("Jigsaw matrix shape : ({},{}) ".format(x_j, y_j))

        raise ValueError("The Jigsaw configuration is too big for the image."
                         "Please decrease the distance between crops but"
                         "better the crops size")

    initialJigsawXPosition, initialJigsawYPosition = determine_upper_left_jigsaw_position(p_image_shape,
                                                                                          (x_j, y_j))

    # Now, we randomly determine the crop we want to compare with crop number 5
    # (the one in the middle)
    # We get the coordinates of the central crop

    crop_4_pos= estimate_crop_position(4,initialJigsawXPosition, initialJigsawYPosition,
                           x_c, y_c, d_c)

    random_crop_position = 4

    while(random_crop_position ==4):
        random_crop_position = int(random.random() * 8)

    # We get the coordinates of the randomly chosen crop
    crop_random_pos = estimate_crop_position(random_crop_position,initialJigsawXPosition,
                                             initialJigsawYPosition,
                                             x_c, y_c, d_c)

    return crop_4_pos, crop_random_pos, random_crop_position

def perform_crop(p_crop_position, p_original_image, p_crop_size=(112, 112)):

    # print("crop positions : ", p_crop_position)

    crop_top = p_crop_position[2]
    crop_left = p_crop_position[0]
    crop_height = p_crop_position[3] - p_crop_position[2]
    crop_width = p_crop_position[1] - p_crop_position[0]

    crop = TF.resized_crop(p_original_image, crop_top, crop_left,
                           crop_height, crop_width, p_crop_size)

    return crop


def put_squares_to_pil_image(p_pil_image,
                             p_first_square,
                             p_second_square):

    opencv_image = np.array(p_pil_image)

    # print("starting point : ", (p_first_square[0], p_first_square[2]))
    # print("ending point : ", (p_first_square[1], p_first_square[3]))

    cv2.rectangle(opencv_image, (p_first_square[0], p_first_square[2]),
                  (p_first_square[1], p_first_square[3]), (255,0,0), 2)

    cv2.rectangle(opencv_image, (p_second_square[0], p_second_square[2]),
                  (p_second_square[1], p_second_square[3]), (0,255,0), 2)

    im_pil = Image.fromarray(opencv_image)

    return im_pil


class jigsaw_dataset(data.Dataset):

    def __init__(self, dataset, shuffle, batch_size, num_workers):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([transforms.Resize(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485,0.456, 0.406],
                                                                  [0.229,0.224, 0.225])])

    def __getitem_for_debugging__(self, index):

        idx = index % len(self.dataset)
        img, _ = self.dataset[idx]

        # print(type(img))

        imgShape = img.size

        # For the moment I set the crop shape as a fraction of the image shape
        crop_shape = (int(imgShape[0]/4), int(imgShape[1]/4))


        # Also the distance between crops
        crops_distance = int(imgShape[0] / 8)


        centralCrop_pos, randomCrop_pos, randomCropPosition = determine_the_two_crops_positions(imgShape,
                                                                      crop_shape,
                                                                      crops_distance)

        centralCrop = perform_crop(centralCrop_pos, img, crop_shape)
        randomCrop = perform_crop(randomCrop_pos, img, crop_shape)


        # I add the image add the end just for debugging purposes
        crops = [self.transform(centralCrop), self.transform(randomCrop)]

        img_squared = put_squares_to_pil_image(img,
                                 centralCrop_pos,
                                 randomCrop_pos)

        # print(crops)
        print(randomCropPosition)

        # return torch.stack(crops, dim=0), torch.LongTensor([randomCropPosition]), self.transform(img)

        return self.transform(centralCrop), self.transform(randomCrop), self.transform(img_squared)

    def __getitem__(self, index):

        idx = index % len(self.dataset)
        img, _ = self.dataset[idx]

        # print(type(img))

        imgShape = img.size

        # For the moment I set the crop shape as a fraction of the image shape
        crop_shape = (int(imgShape[0]/4), int(imgShape[1]/4))


        # Also the distance between crops
        crops_distance = int(imgShape[0] / 8)


        centralCrop_pos, randomCrop_pos, randomCropPosition = determine_the_two_crops_positions(imgShape,
                                                                      crop_shape,
                                                                      crops_distance)

        centralCrop = self.transform(perform_crop(centralCrop_pos, img, crop_shape))
        randomCrop = self.transform(perform_crop(randomCrop_pos, img, crop_shape))

        # I add the image add the end just for debugging purposes
        crops = [centralCrop, randomCrop]

        # return self.transform(centralCrop), self.transform(randomCrop), self.transform(img)

        return centralCrop, randomCrop, torch.LongTensor([randomCropPosition])

    def __len__(self):
        return 1000

