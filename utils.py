import numpy as np
import torch
import os
import math
import cv2

def tensor2np(input_tensor, rgb_range=255):
    """

    :param input_tensor: C*H*W tenosr input
    :param rgb_range: data range you want to use while training (No negative range!)
    :param imtype: save numpy data type
    :return: H*W*C size numpy output
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError('Input is not tensor')
    elif len(input_tensor.size()) != 3:
        raise ValueError('input size must be C*H*W')

    tensor = input_tensor.data.mul(255 / rgb_range).round()
    image_numpy = tensor.byte().permute(1, 2, 0).cpu().numpy()

    return image_numpy

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

# def tensor2np(input_tensor, rgb_range=255, imtype=np.uint8):
#     """
#
#     :param input_tensor: C*H*W tenosr input
#     :param rgb_range: data range you want to use while training (No negative range!)
#     :param imtype: save numpy data type
#     :return: H*W*C size numpy output
#     """
#     if not isinstance(input_tensor, torch.Tensor):
#         raise TypeError('Input is not tensor')
#     elif len(input_tensor.size()) != 3:
#         raise ValueError('input size must be C*H*W')
#
#     tensor = input_tensor.cpu().float().mul_(255 / rgb_range)
#     image_numpy = tensor.byte().numpy()
#     image_numpy = image_numpy.transpose(1, 2, 0)
#     image_numpy = np.clip(image_numpy, 0, 255)
#
#     return image_numpy.astype(imtype)

def np2tensor(img, rgb_range):
    """
    numpy to tensor
    :param img: H*W*C size numpy input
    :param rgb_range: data range you want to use while training (No negative range!)
    :return: C*H*W tenosr output
    """
    np_transpose = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor


def inverse_normalize(y, mean, std):
    """

    :param y: normalized input tensor, size: N*C*H*W
    :param mean: type: list, len(mean) == C
    :param std: type: list, len(std) == C
    :return: inverse normalized output
    """
    if not isinstance(y, torch.Tensor):
        raise TypeError('Input is not tensor')

    if not (isinstance(mean, list) and isinstance(std, list)):
        raise TypeError('mean and std must be list')

    if len(mean) != y.size()[1] or len(std) != y.size()[1]:
        raise ValueError('lengths of mean and std must be equal to channel of input')

    x = y.new(*y.size())
    for i in range(x.size()[1]):
        x[:, i, :, :] = y[:, i, :, :] * std[i] + mean[i]

    return x


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rgb2ycbcr(rgb):
    """
    the same as matlab rgb2ycbcr
    :param rgb: input [0, 255] or [0, 1]
    :return: output [0, 255] or [0, 1]
    """
    in_img_type = rgb.dtype
    rgb = rgb.astype(np.float64)
    if in_img_type != np.uint8:
        rgb *= 255.
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0]*shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    # ycbcr = np.clip(ycbcr, 0, 255)
    if in_img_type == np.uint8:
        ycbcr = ycbcr.round()
    else:
        ycbcr /= 255.

    return ycbcr.reshape(shape).astype(in_img_type)

def ycbcr2rgb(ycbcr):
    """
    the same as matlab ycbcr2rgb
    :param rgb: input [0, 255] or [0, 1]
    :return: output [0, 255] or [0, 1]
    """
    in_img_type = ycbcr.dtype
    ycbcr = ycbcr.astype(np.float64)
    if in_img_type != np.uint8:
        ycbcr *= 255.
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])

    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0]*shape[1], 3))

    rgb = np.copy(ycbcr)
    rgb[:, 0] -= 16.
    rgb[:, 1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    rgb = np.clip(rgb, 0, 255)
    if in_img_type == np.uint8:
        rgb = rgb.round()
    else:
        rgb /= 255.

    return rgb.reshape(shape).astype(in_img_type)

def ssim(img1, img2):

    C1 = (0.01 * 255) **2
    C2 = (0.03 * 255) **2
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())


    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def calc_ssim(img1, img2):
    """
    calculate SSIM the same as matlab
    input [0, 255]

    :return:
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimension')

def calc_PSNR(pred, gt):
    """
    calculate PSNR the same as matlab
    input [0, 255] float
    :param pred:
    :param gt:
    :return:
    """
    if not pred.shape == gt.shape:

        raise ValueError('Input images must have the same dimensions.')
    if pred.ndim != 2:
        raise ValueError('Input images must be H*W.')

    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

if __name__ == '__main__':
    import cv2
    path = '/media/luo/data/data/Set5/Set5/baby_GT.bmp'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ycbcr = rgb2ycbcr(img)
    # ycbcr1 = rgb2ycbcr1(img)
    rgb = ycbcr2rgb(ycbcr)
    # rgb1 = ycbcr2rgb(ycbcr1)
    print(ycbcr)


