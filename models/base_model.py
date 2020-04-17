
import torch
import os
import utils
from torch.nn import init
from utils import tensor2np, calc_ssim, rgb2ycbcr, calc_PSNR
import cv2
import numpy as np
import imageio

class BaseModel():

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')


        """save model path"""
        self.save_dir = os.path.join('checkpoints', opt.SR_name)
        utils.mkdir(self.save_dir)

        """save results path"""
        self.result_dir = os.path.join('results', opt.SR_name)
        _, testset = os.path.split(opt.testroot)
        self.result_dir = os.path.join(self.result_dir, testset)
        utils.mkdir(self.result_dir)

        """save log path"""
        log_dir = os.path.join(self.save_dir, '%s_log.txt' % opt.SR_name)

        open_type = 'a' if os.path.exists(log_dir) else 'w'
        self.log_file = open(log_dir, open_type)
        self.log_file.write('\n')
        for arg in vars(opt):
            self.log_file.write('{}:{}\n'.format(arg, getattr(opt, arg)))
        self.log_file.write('\n')

    def eval_initialize(self, opt):

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        """save results path"""
        self.result_dir = os.path.join('results', opt.SR_name)
        _, testset = os.path.split(opt.testroot)
        self.result_dir = os.path.join(self.result_dir, testset)
        utils.mkdir(self.result_dir)


    def set_mode(self, train):
        if train:
            self.model.train()
        else:
            self.model.eval()


    def save_model(self, epoch, name):

        save_filename = '%s_%d.pth' % (name, epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        net = self.model
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            print(save_filename)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)
            print(save_filename)


    def load_model(self, epoch, name):

        load_filename = '%s_%d.pth' % (name, epoch)
        load_path = os.path.join(self.save_dir, load_filename)
        net = self.model
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        net.load_state_dict(state_dict)

    def save_image(self, img_tensor, save_name):
        img_np = tensor2np(img_tensor, self.opt.rgb_range)
        imageio.imsave(save_name, img_np)
        # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(save_name, img_np)

    def comput_PSNR_SSIM(self, pred, gt, shave_border=0):

        if isinstance(pred, torch.Tensor):
            pred = tensor2np(pred, self.opt.rgb_range)
            pred = pred.astype(np.float32)

        if isinstance(gt, torch.Tensor):
            gt = tensor2np(gt, self.opt.rgb_range)
            gt = gt.astype(np.float32)


        height, width = pred.shape[:2]
        pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]

        if pred.shape[2] == 3 and gt.shape[2] == 3:
            pred_y = rgb2ycbcr(pred)[:, :, 0]
            gt_y = rgb2ycbcr(gt)[:, :, 0]
        elif pred.shape[2] == 1 and gt.shape[2] == 1:
            pred_y = pred[:, :, 0]
            gt_y = gt[:, :, 0]
        else:
            raise ValueError('Input or output channel is not 1 or 3!')

        psnr_ = calc_PSNR(pred_y, gt_y)
        ssim_ = calc_ssim(pred_y, gt_y)

        return psnr_, ssim_



def init_weights(net, init_type):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)