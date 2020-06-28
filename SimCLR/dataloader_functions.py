import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

# From https://arxiv.org/pdf/2002.05709.pdf
# In this work, we sequentially apply three simple augmentations:
# random cropping followed by resize back to the original size,
# random color distorions, and random Gaussian blur.
# (the combination of random crop and color distorion
# is crucial to achieve a good performance).


# Gaussian Kernel Options
# https://discuss.pytorch.org/t/gaussian-kernel-layer/37619
# https://discuss.pytorch.org/t/gaussian-filter-for-images/20597
# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/2


# From https://courses.cs.washington.edu/courses/cse446/19au/section9.html

gaussianKernel = torch.tensor([[1., 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]]) /16.0

gaussianConv = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
# with torch.no_grad():
#     gaussianConv.weight =gaussianKernel
gaussianConv.weight.data [:] = gaussianKernel

print(gaussianConv.weight)

# Now, for defining a custom operation as a pytorch transform :
# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/11

class GaussianBlur(object):

    def __call__(self, img):

        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
        # print(img.shape)

        return gaussianConv(img)


class simclrDataset(data.Dataset):

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
            transforms.ToTensor(),
            GaussianBlur()
        ])

        # self.transform = transforms.Compose([transforms.Resize((256, 256))])


    def __getitem__(self, index):

        idx = index % len(self.dataset)

        img, _ = self.dataset[idx]

        first_transform = self.transform(img)
        second_transform = self.transform(img)

        img_shape = first_transform.shape
        # print("shape ", img_shape)

        return first_transform.reshape(img_shape[1],img_shape[2],img_shape[3]), \
               second_transform.reshape(img_shape[1],img_shape[2],img_shape[3])

    def __len__(self):
        return int(( len(self.dataset) / 4)/ self.batch_size )


def rotation_transform(image, angle):

    r_image = TF.rotate(image, angle)

    return r_image

class rotationSiMCLR(data.Dataset):

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

    def __getitem__(self, index):

        idx = index % len(self.dataset)

        img, _ = self.dataset[idx]

        rotation = int(random.random() * 4)

        # if rotation == 0:



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