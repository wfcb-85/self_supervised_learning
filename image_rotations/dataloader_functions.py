import torch
import torch.utils.data as data
import torchvision
# import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.dataloader import default_collate

import numpy as np
import random

from PIL import Image

import torchnet as tnt

import torchvision.transforms.functional as TF

def collate_fn_alter(batch):

    # From https://github.com/yunjey/pytorch-tutorial/blob/master/
    # tutorials/03-advanced/image_captioning/data_loader.py

    print(" MAMA "* 20)

    images, labels  = zip(*batch)

    images = torch.stack(images, 0)

    targets = torch.stack(labels, 0)

    return images, targets



def _collate_fun(batch):

        # print("batch", len(batch))
        # print("batch elem size : ", batch[0])

        # print("batch ", batch)

        """
        Before the default_collate, each batch is a list containing each of the #batch_size
        components. Where each component is a tuple containing two tensors,

        The first one, containing the 4 rotations obtained from a single image.
            Its shape is [#Number of Rotations, channels, height, width]
        The second containing the rotation labels. Its shape is [4]


        The output of the function -default_collate()- is a list containing two tensors
        The first one, of size [batch_size, #number of rotations, channels, height, width]

        And the second, which contains the labels, of shape [batch_size, #Number of rotations]

        The post-processing of this functions creates 2 tensors, one of the shape
        [batch_size * rotations,  channels, weight, height], a


        :param batch:
        :return:
        """

        # print("len before ", len(batch))
        # print("element size ", len(batch[0]))
        # print("inside ", batch[0][0].size())
        # print("inside lab", batch[0][1].size())

        batch = default_collate(batch)
        assert(len(batch)==2)

        # print("inside after ", batch[0].size())
        # print("inside after  lab", batch[1].size())

        batch_size, rotations, channels, height, width = batch[0].size()
        batch[0] = batch[0].view([batch_size * rotations, channels, height, width])
        # batch[1] = batch[1].view([batch_size, rotations])
        batch[1] = batch[1].view([batch_size * rotations])


        return batch

def rotate_image(img, rot):

    """
    https://github.com/gidariss/FeatureLearningRotNet/blob/master/dataloader.py

    """

    if rot == 0:
        return img,

    elif rot == 90:
        return np.flipud(np.transpose(img, (1, 0, 2)))

    elif rot == 180:
        return np.fliplr(np.flipud(img))

    elif rot == 270:
        return np.transpose(np.flipud(img), (1, 0, 2))

    else:
        raise ValueError("rotation should be one of 0, 90, 180, or 270 degrees")

class Denormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def rotation_transform(image, angle):

    r_image = TF.rotate(image, angle)

    return r_image


class wc_rotation_DataSet(data.Dataset):

    def __init__(self, dataset, shuffle,  batch_size, num_workers):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

        self.resize_pil_image = transforms.Compose([
            transforms.Resize((256, 256))
            # transforms.ToPILImage()
        ])

        # self.transform = transforms.Compose([transforms.Resize((256, 256))])


    def __getitem__(self, index):

        idx = index % len(self.dataset)

        img, _ = self.dataset[idx]

        # print(img.size)
        # print(type(img))
        # print(type(self.resize_pil_image(img)))

        rotated_imgs = [
            self.transform(img),

            # What we do here is to first resize the image for it to be square. Otherwise, there would
            # be some visible borders when the rotation is performed.

            torchvision.transforms.ToTensor()(rotation_transform(self.resize_pil_image(img), 90)),
            self.transform(rotation_transform(img, 180)),
            torchvision.transforms.ToTensor()(rotation_transform(self.resize_pil_image(img), 270))
        ]
        rotation_labels = torch.LongTensor([0, 1, 2, 3])

        # print(rotated_imgs)
        # print(rotation_labels)

        return torch.stack(rotated_imgs, dim=0), rotation_labels

    # def _collate_fun(batch):
    #
    #     batch = default_collate(batch)
    #     assert(len(batch)==2)
    #     batch_size, rotations, channels, height, width = batch[0].size()
    #     batch[0] = batch[0].view([batch_size * rotations, channels, height, width])
    #     batch[1] = batch[1].view([batch_size, rotations])
    #     return batch

    def __len__(self):
        return int(( len(self.dataset) / 4)/ self.batch_size )


class DataLoader(object):

    def __init__(self, dataset, batch_size=1,
                 epoch_size =None, num_workers=0, shuffle=True):

        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset.mean_pix = 125
        self.dataset.std_pix = 30

        mean_pix = self.dataset.mean_pix
        std_pix = self.dataset.std_pix

        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean_pix, std=std_pix)
        # ])

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8)
        ])

    def get_iterator(self, epoch=0):

        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)

        def _load_function(idx):

            """
            It recovers the image with id 'idx', then it creates 4 transformed images from it,
            where each transformation corresponds to one rotation. Finally, it return the
            transformed images along with the labels of the rotation.

            :param idx:
            :return:
            """

            idx = idx % len(self.dataset)
            img0, _ = self.dataset[idx]


            rotated_imgs = [
                self.transform(img0),
                self.transform(rotate_image(img0, 90)),
                self.transform(rotate_image(img0, 180)),
                self.transform(rotate_image(img0, 270))
            ]
            rotation_labels = torch.LongTensor([0, 1, 2, 3])

            return torch.stack(rotated_imgs, dim=0), rotation_labels

        def _collate_fun(batch):

            batch = default_collate(batch)
            assert(len(batch)==2)
            batch_size, rotations, channels, height, width = batch[0].size()
            batch[0] = batch[0].view([batch_size * rotations, channels, height, width])
            batch[1] = batch[1].view([batch_size, rotations])
            return batch

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=_load_function)

        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=_collate_fun,
                                           num_workers=self.num_workers,
                                           shuffle=self.shuffle)

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size