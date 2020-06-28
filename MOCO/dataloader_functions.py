import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class mocoDataset(data.Dataset):

    def __init__(self, dataset, shuffle,  batch_size, num_workers):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            # The size of the random crop must be set as a parameter
            transforms.RandomCrop((64,64)),
            transforms.Resize((256, 256)),
            transforms.ColorJitter(),
            transforms.ToTensor()
        ])

        # self.transform = transforms.Compose([transforms.Resize((256, 256))])


    def __getitem__(self, index):

        idx = index % len(self.dataset)

        img, _ = self.dataset[idx]

        first_transform = self.transform(img)
        second_transform = self.transform(img)

        img_shape = first_transform.shape

        return first_transform.reshape(img_shape[0],img_shape[1],img_shape[2]), \
               second_transform.reshape(img_shape[0],img_shape[1],img_shape[2])

    def __len__(self):
        return int(( len(self.dataset) / 4)/ self.batch_size )


def rotation_transform(image, angle):

    r_image = TF.rotate(image, angle)

    return r_image
