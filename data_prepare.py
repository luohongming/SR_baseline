
"""
 Speed of reading one image : cv2.imread + cvtColor(BGR2RGB)  0.07
                              scipy.msic.imread               0.1
                              numpy load                      0.002

 In order to speed up the training process. We plan to save images data as .npy file. Then load them.
"""
import cv2
import os
from utils import mkdir
import numpy as np
from matlab_imresize import imresize

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
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

def save_image2npy(path, save_path, sr_fator):

    images = make_dataset(path)
    input_save_path = os.path.join(save_path, 'x%d' % (sr_fator))
    target_save_path = os.path.join(save_path, 'origin')
    mkdir(input_save_path)
    mkdir(target_save_path)

    for img_name in images:
        print(img_name)
        _, name = os.path.split(img_name)
        name, _ = name.split('.')
        target_name = '%s.npy' % name
        input_name = '%s.npy' % name
        input_save_name = os.path.join(input_save_path, input_name)
        target_save_name = os.path.join(target_save_path, target_name)

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_down = imresize(img, scalar_scale=1/sr_fator)
        print(img.shape, img_down.shape)
        np.save(target_save_name, img)
        np.save(input_save_name, img_down)


if __name__ == '__main__':
    path = '/media/luo/data/data/super-resolution/Train_291'
    save_path = '/media/luo/data/data/super-resolution/Train_291_npy'

    # save_image2npy(path, save_path, 2)
    a = np.load('/media/luo/data/data/super-resolution/Train_291_npy/x2/2092_00.npy')
    b = np.load('/media/luo/data/data/super-resolution/Train_291_npy/origin/2092_00.npy')
    print(a.shape, b.shape)
