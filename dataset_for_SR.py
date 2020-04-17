
import torch.utils.data as data
import h5py
import torch
import random
import utils
import numpy as np
from matlab_imresize import imresize
import os
import cv2


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy', '.NPY'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def make_dataset_from_npy(target_dir, input_dir):
    images_input = []
    images_target = []
    assert os.path.isdir(target_dir), '%s is not a valid directory' % target_dir

    for root, _, fnames in sorted(os.walk(target_dir)):
        for fname in fnames:
            if is_image_file(fname):
                target_path = os.path.join(target_dir, fname)
                images_target.append(target_path)
                input_path = os.path.join(input_dir, fname)
                images_input.append(input_path)

    return images_input, images_target


class DatasetFromHdf5Train(data.Dataset):
    def __init__(self, opt, rgb=True, input_up=False):
        super(DatasetFromHdf5Train, self).__init__()
        self.sr_factor = opt.sr_factor
        self.file_path = opt.trainroot
        hf = h5py.File(opt.file_path, 'r')
        self.input = hf['input']
        self.target = hf['target']
        self.patch_size = opt.patch_size
        self.rgb_range = opt.rgb_range
        self.rgb = rgb
        self.input_up = input_up

    def __getitem__(self, index):
        input_ = self.input[index, :, :, :]
        target_ = self.target[index, :, :, :]

        subim_in, subim_tar = get_patch(input_, target_, self.patch_size, self.sr_factor)

        if not self.rgb:
            subim_in = utils.rgb2ycbcr(subim_in)
            subim_tar = utils.rgb2ycbcr(subim_tar)
            subim_in = np.expand_dims(subim_in[:, :, 0], 2)
            subim_tar = np.expand_dims(subim_tar[:, :, 0], 2)

        if self.input_up:
            subim_bic = imresize(subim_in, scalar_scale=self.sr_factor)
            subim_in = utils.np2tensor(subim_in, self.rgb_range)
            subim_tar = utils.np2tensor(subim_tar, self.rgb_range)
            subim_bic = utils.np2tensor(subim_bic, self.rgb_range)
            return {'input': subim_in, 'target': subim_tar, 'input_up': subim_bic}

        subim_in = utils.np2tensor(subim_in, self.rgb_range)
        subim_tar = utils.np2tensor(subim_tar, self.rgb_range)

        return {'input': subim_in, 'target': subim_tar}


    def __len__(self):

        return self.input.shape[0]


class DatasetFromHdf5Test(data.Dataset):
    def __init__(self, opt, rgb=True, input_up=False):
        super(DatasetFromHdf5Test, self).__init__()
        self.sr_factor = opt.sr_factor
        self.file_path = opt.testroot
        hf = h5py.File(opt.file_path, 'r')
        self.input = hf['input']
        self.target = hf['target']
        self.rgb_range = opt.rgb_ragne
        self.rgb = rgb
        self.input_up = input_up

    def __getitem__(self, index):
        input_ = self.input[index, :, :, :]
        target_ = self.input[index, :, :, :]

        if not self.rgb:
            input_ = utils.rgb2ycbcr(input_)
            target_ = utils.rgb2ycbcr(target_)

        if self.input_up:
            input_bic = imresize(input_, scalar_scale=self.sr_factor).round()
            input_ = utils.np2tensor(input_, self.rgb_range)
            target_ = utils.np2tensor(target_, self.rgb_range)
            input_bic_ = utils.np2tensor(input_bic, self.rgb_range)
            return {'input': input_, 'target': target_, 'input_up': input_bic_}

        input_ = utils.np2tensor(input_, self.rgb_range)
        target_ = utils.np2tensor(target_, self.rgb_range)
        return {'input': input_, 'target': target_}

    def __len__(self):
        return self.input.shape[0]


class DatasetFromImageTrain(data.Dataset):
    def __init__(self, opt, rgb=True, input_up=False):
        super(DatasetFromImageTrain, self).__init__()
        self.sr_factor = opt.sr_factor
        self.file_path = opt.trainroot
        self.patch_size = opt.patch_size
        self.images_path = make_dataset(self.file_path)
        self.rgb_range = opt.rgb_ragne
        self.rgb = rgb
        self.input_up = input_up

    def __getitem__(self, index):
        img_path = self.images_path[index]
        target_ = cv2.imread(img_path)
        input_ = imresize(target_, scalar_scale=1 / self.sr_factor)

        subim_in, subim_tar = get_patch(input_, target_, self.patch_size, self.sr_factor)

        if not self.rgb:
            subim_in = utils.rgb2ycbcr(subim_in)
            subim_tar = utils.rgb2ycbcr(subim_tar)
            subim_in = np.expand_dims(subim_in[:, :, 0], 2)
            subim_tar = np.expand_dims(subim_tar[:, :, 0], 2)

        if self.input_up:
            subim_bic = imresize(subim_in, scalar_scale=self.sr_factor)
            subim_in = utils.np2tensor(subim_in, self.rgb_range)
            subim_tar = utils.np2tensor(subim_tar, self.rgb_range)
            subim_bic = utils.np2tensor(subim_bic, self.rgb_range)
            return {'input': subim_in, 'target': subim_tar, 'input_up': subim_bic}

        subim_in = utils.np2tensor(subim_in, self.rgb_range)
        subim_tar = utils.np2tensor(subim_tar, self.rgb_range)

        return {'input': subim_in, 'target': subim_tar}

    def __len__(self):
        return len(self.images_path)


class DatasetFromImageTest(data.Dataset):
    def __init__(self, opt, rgb=True, input_up=False):
        super(DatasetFromImageTest, self).__init__()
        self.sr_factor = opt.sr_factor
        self.file_path = opt.trainroot
        self.patch_size = opt.patch_size

        self.images_path = make_dataset(self.file_path)
        self.rgb_range = opt.rgb_ragne
        self.rgb = rgb
        self.input_up = input_up

    def __getitem__(self, index):
        img_path = self.images_path[index]
        target_ = cv2.imread(img_path)
        input_ = imresize(target_, scalar_scale=1 / self.sr_factor)

        if not self.rgb:
            input_ = utils.rgb2ycbcr(input_)
            target_ = utils.rgb2ycbcr(target_)

        if self.input_up:
            input_bic = imresize(input_, scalar_scale=self.sr_factor).round()
            input_ = utils.np2tensor(input_, self.rgb_range)
            target_ = utils.np2tensor(target_, self.rgb_range)
            input_bic_ = utils.np2tensor(input_bic, self.rgb_range)
            return {'input': input_, 'target': target_, 'input_up': input_bic_}

        input_ = utils.np2tensor(input_, self.rgb_range)
        target_ = utils.np2tensor(target_, self.rgb_range)
        return {'input': input_, 'target': target_}

    def __len__(self):
        return len(self.images_path)

class DatasetFromNpyTrain(data.Dataset):
    def __init__(self, opt, rgb=True, input_up=False):
        super(DatasetFromNpyTrain, self).__init__()
        self.sr_factor = opt.sr_factor
        self.target_file_path = os.path.join(opt.trainroot, 'origin')
        self.input_file_path = os.path.join(opt.trainroot, 'x%d' % self.sr_factor)
        self.patch_size = opt.patch_size
        # self.input_path = make_dataset(self.input_file_path)
        # self.target_path = make_dataset(self.target_file_path)
        self.input_path, self.target_path = make_dataset_from_npy(self.target_file_path, self.input_file_path)
        self.rgb_range = opt.rgb_range
        self.rgb = rgb
        self.input_up = input_up

    def __getitem__(self, index):
        # print(index)
        input_ = np.load(self.input_path[index])
        target_ = np.load(self.target_path[index])
        subim_in, subim_tar = get_patch(input_, target_, self.patch_size, self.sr_factor)

        if not self.rgb:
            subim_in = utils.rgb2ycbcr(subim_in)
            subim_tar = utils.rgb2ycbcr(subim_tar)
            subim_in = np.expand_dims(subim_in[:, :, 0], 2)
            subim_tar = np.expand_dims(subim_tar[:, :, 0], 2)


        if self.input_up:
            subim_bic = imresize(subim_in, scalar_scale=self.sr_factor)
            subim_in = utils.np2tensor(subim_in, self.rgb_range)
            subim_tar = utils.np2tensor(subim_tar, self.rgb_range)
            subim_bic = utils.np2tensor(subim_bic, self.rgb_range)
            return {'input': subim_in, 'target': subim_tar, 'input_up': subim_bic}

        subim_in = utils.np2tensor(subim_in, self.rgb_range)
        subim_tar = utils.np2tensor(subim_tar, self.rgb_range)

        return {'input': subim_in, 'target': subim_tar}

    def __len__(self):
        return len(self.input_path)


class DatasetFromNpyTest(data.Dataset):
    def __init__(self, opt, rgb=True, input_up=False):
        super(DatasetFromNpyTest, self).__init__()
        self.sr_factor = opt.sr_factor
        self.target_file_path = os.path.join(opt.testroot, 'origin')
        self.input_file_path = os.path.join(opt.testroot, 'x%d' % self.sr_factor)
        self.patch_size = opt.patch_size
        self.input_path, self.target_path = make_dataset_from_npy(self.target_file_path, self.input_file_path)
        self.rgb_range = opt.rgb_range
        self.rgb = rgb
        self.input_up = input_up

    def __getitem__(self, index):
        input_ = np.load(self.input_path[index])
        target_ = np.load(self.target_path[index])

        if not self.rgb:
            input_ = utils.rgb2ycbcr(input_)
            target_ = utils.rgb2ycbcr(target_)

        if self.input_up:
            input_bic = imresize(input_, scalar_scale=self.sr_factor).round()
            input_ = utils.np2tensor(input_, self.rgb_range)
            target_ = utils.np2tensor(target_, self.rgb_range)
            input_bic_ = utils.np2tensor(input_bic, self.rgb_range)
            return {'input': input_, 'target': target_, 'input_up': input_bic_}

        input_ = utils.np2tensor(input_, self.rgb_range)
        target_ = utils.np2tensor(target_, self.rgb_range)
        return {'input': input_, 'target': target_}

    def __len__(self):
        return len(self.input_path)



def get_patch(img_in, img_tar, patch_size, scale):
    ih, iw = img_in.shape[:2]
    # p = scale
    ip = patch_size
    tp = ip * scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar

