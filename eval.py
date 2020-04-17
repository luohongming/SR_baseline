import argparse
import torch
import torch.backends.cudnn as cudnn
from models.IDN import IDNModel
from models.RCAN import RCANModel
from models.EDSR import EDSRModel
from models.RDN import RDNModel
from dataset_for_SR import DatasetFromNpyTest
from torch.utils.data import DataLoader
import os
import numpy as np



parser = argparse.ArgumentParser(description='Super resolution baseline')
# parser.add_argument('--trainroot', default='/media/luo/data/data/super-resolution/DIV2K_train_HR_augment_npy', type=str, help='Train dataset')
parser.add_argument('--testroot', default='/media/luo/data/data/super-resolution/testsets/Set5_npy', type=str, help='Test dataset')
parser.add_argument('--seed', type=int, default=56, help='random seed to use. Default=123')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
# parser.add_argument('--patch_size', type=int, default=48, help='Patch size')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--sr_factor", default=2, type = int, help="Super resolution scale")
# parser.add_argument("--nEpochs", type=int, default=300, help="Number of epochs to train for")
# parser.add_argument("--save_epoch", type=int, default=5, help="Number of saving epoch")
parser.add_argument("--epoch", type=int, default=0, help="The starting epoch, if epoch = 0 new training process "
                                                         "if epoch > 0, continue training")
parser.add_argument("--rgb_range", type=int, default=255, help="Data range")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate, Default=0.1")
parser.add_argument('--SR_name', type=str, default='EDSR',
                    help='name of SR model. It decides where to store models')



def main():
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    # set gpu_ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    cudnn.benchmark = True
    print(opt)

    if opt.SR_name == 'IDN':
        model = IDNModel()
        test_dataset = DatasetFromNpyTest(opt, rgb=False, input_up=True)

    elif opt.SR_name == 'RCAN':
        model = RCANModel()
        test_dataset = DatasetFromNpyTest(opt, rgb=True)

    elif opt.SR_name == 'EDSR':
        model = EDSRModel()
        test_dataset = DatasetFromNpyTest(opt, rgb=True)

    elif opt.SR_name == 'RDN':
        model = RDNModel()
        test_dataset = DatasetFromNpyTest(opt, rgb=True)
    else:
        raise NotImplementedError('%s is not supported!' % opt.SR_name)

    print(len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=1)

    model.eval_initialize(opt)

    model.load_model(opt.epoch, opt.SR_name)

    model.set_mode(train=False)
    average_psnr = []
    average_ssim = []
    for i, data in enumerate(test_loader):
        model.set_eval_input(data)
        outputs = model.eval()
        psnr_, ssim_ = model.comput_PSNR_SSIM(outputs['output'], outputs['target'], shave_border=opt.sr_factor)
        average_psnr.append(psnr_)
        average_ssim.append(ssim_)

        img_type = ['input', 'output', 'target']

        for name, img in zip(img_type, outputs):
            save_name = os.path.join(model.result_dir, '%d_%s.png' % (i, name))
            model.save_image(outputs[img], save_name)

    average_psnr = np.average(average_psnr)
    average_ssim = np.average(average_ssim)
    log = 'Epoch %d: Average psnr: %f , ssim: %f \n' % (opt.epoch, average_psnr, average_ssim)
    print(log)




if __name__ == '__main__':
    main()
