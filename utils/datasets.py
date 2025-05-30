import h5py
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import pandas as pd
import torchvision.transforms.functional as TF
import random

def array_to_img(x, data_format='channels_last'):
    # 数据类型检查
    if not isinstance(x, np.ndarray):
        raise ValueError("输入必须是 NumPy 数组")

    if data_format == 'channels_first':
        x = np.transpose(x, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    elif data_format != 'channels_last':
        raise ValueError("data_format 必须是 'channels_first' 或 'channels_last'")
    x = np.clip(x, 0, 255).astype('uint8')
    return Image.fromarray(x)


def img_to_array(image, data_format='channels_last', dtype='float32'):
    if isinstance(image, Image.Image):
        x = np.array(image, dtype=dtype)
    elif isinstance(image, np.ndarray):
        x = image.astype(dtype)
    else:
        raise ValueError("输入必须是 PIL.Image 或 NumPy 数组")

    if x.ndim == 2:
        x = np.expand_dims(x, axis=-1)

    if data_format == 'channels_first':
        x = np.transpose(x, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    elif data_format != 'channels_last':
        raise ValueError("data_format 必须是 'channels_first' 或 'channels_last'")
    return x


class OralDataset(Dataset):
    def __init__(self, train_test_id, image_path, train_test_split_file,
                 train=True, attribute=None, transform=None, num_classes=None):

        self.resize = [640, 448]
        self.std = [0, 0, 0]
        self.padding_value = 0
        self.flip_hor = 0.5
        self.rotate = 0.3
        self.angle = 10
        self.crop_range = [0.75, 1.0]

        self.train_test_id = train_test_id
        self.image_path = image_path
        self.train = train
        self.attr_types = ['OLP', 'OLK', 'OBU']
        self.attribute = attribute

        self.transform = transform
        self.num_classes = num_classes

        self.mask_ind = pd.read_csv(train_test_split_file, index_col=0)
        self.mask_ind = pd.DataFrame(self.mask_ind)

        if self.attribute is not None and self.attribute != 'all':
            print('mask type: ', self.mask_attr, 'train_test_id.shape: ', self.train_test_id.shape)

        if self.train:
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] == 'train'].ID.values
            print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
        else:
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] == 'val'].ID.values
            print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
        self.n = self.train_test_id.shape[0]

    def __len__(self):
        return self.n

    def transform_fn(self, image, mask):
        image = array_to_img(image, data_format="channels_last")
        mask_pil_array = [None] * mask.shape[-1]
        for i in range(mask.shape[-1]):
            mask_pil_array[i] = array_to_img(mask[:, :, i], data_format="channels_last")

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            for i in range(mask.shape[-1]):
                mask_pil_array[i] = TF.hflip(mask_pil_array[i])

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            for i in range(mask.shape[-1]):
                mask_pil_array[i] = TF.vflip(mask_pil_array[i])

        # Random to_grayscale
        if random.random() > 0.6:
            image = TF.to_grayscale(image, num_output_channels=3)

        angle = random.randint(0, 90)
        translate = (random.uniform(0, 100), random.uniform(0, 100))
        scale = random.uniform(0.5, 2)
        shear = random.uniform(0, 0)
        image = TF.affine(image, angle, translate, scale, shear)
        for i in range(mask.shape[-1]):
            mask_pil_array[i] = TF.affine(mask_pil_array[i], angle, translate, scale, shear)

        # Random adjust_brightness
        image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.9, 1.1))

        # Random adjust_saturation
        image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.9, 1.1))

        # Transform to tensor
        image = img_to_array(image, data_format="channels_last")
        for i in range(mask.shape[-1]):
            mask[:, :, i] = img_to_array(mask_pil_array[i], data_format="channels_last")[:, :, 0].astype('uint8')

        image = (image / 255.0).astype('float32')
        mask = (mask / 255.0).astype('uint8')

        return image, mask

    def __getitem__(self, index):
        img_id = self.train_test_id[index]

        image_file = self.image_path + "/%s.h5" % img_id
        img_np = load_image(image_file)
        mask_np = load_mask(self.image_path, img_id)
        if self.train:
            img_np, mask_np = self.transform_fn(img_np, mask_np)
        else:
            img_np = (img_np / 255.0).astype('float32')
            mask_np = (mask_np / 255.0).astype('uint8')
        ind = self.mask_ind[self.mask_ind['ID'] == img_id][self.attr_types].values.astype('uint8')
        return torch.from_numpy(img_np), torch.from_numpy(mask_np), torch.from_numpy(ind[0])


def load_image(image_file):
    f = h5py.File(image_file, 'r')
    img_np = f['image'][()]
    return img_np

def load_mask(image_path, img_id):
    mask_file = image_path + "/%s_all.h5" % (img_id)
    f = h5py.File(mask_file, 'r')
    mask_np = f['image'][()]
    mask_np = mask_np.astype('uint8')
    return mask_np

def make_loader(train_test_id, image_path, args, train, shuffle, train_test_split_file):
    data_set = OralDataset(train_test_id=train_test_id,
                           image_path=image_path,
                           train=train,
                           attribute=args.attribute,
                           transform=None,
                           num_classes=args.num_classes,
                           train_test_split_file=train_test_split_file)

    if train:
        data_loader = DataLoader(data_set,
                                 batch_size=args.batch_size_train,
                                 shuffle=shuffle,
                                 num_workers=48,
                                 pin_memory=torch.cuda.is_available())
    else:
        data_loader = DataLoader(data_set,
                                 batch_size=args.batch_size_val,
                                 shuffle=shuffle,
                                 num_workers=48,
                                 pin_memory=torch.cuda.is_available())

    return data_loader