import argparse
import torch
import torch.backends.cudnn as cudnn
from models.IDN import IDNModel
from models.RCAN import RCANModel
from models.EDSR import EDSRModel
from models.RDN import RDNModel
from dataset_for_SR import DatasetFromNpyTrain, DatasetFromNpyTest
from torch.utils.data import DataLoader
from visualizer import Visualizer
import os
import numpy as np

parser = argparse.ArgumentParser(description='Super resolution baseline')
parser.add_argument('--trainroot', default='/media/luo/data/data/super-resolution/DIV2K_train_HR_augment_npy', type=str, help='Train dataset')
parser.add_argument('--testroot', default='/media/luo/data/data/super-resolution/testsets/Set5_npy', type=str, help='Test dataset')
parser.add_argument('--seed', type=int, default=56, help='random seed to use. Default=123')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--patch_size', type=int, default=48, help='Patch size')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--sr_factor", default=2, type = int, help="Super resolution scale")
parser.add_argument("--nEpochs", type=int, default=300, help="Number of epochs to train for")
parser.add_argument("--save_epoch", type=int, default=5, help="Number of saving epoch")
parser.add_argument("--epoch", type=int, default=0, help="The starting epoch, if epoch = 0 new training process "
                                                         "if epoch > 0, continue training")
parser.add_argument("--rgb_range", type=int, default=255, help="Data range")
parser.add_argument("--lr", type=float, default=1e-4, help="Le arning Rate, Default=0.1")
parser.add_argument('--SR_name', type=str, default='IDN',
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
        train_dataset = DatasetFromNpyTrain(opt, rgb=False, input_up=True)
        test_dataset = DatasetFromNpyTest(opt, rgb=False, input_up=True)

    elif opt.SR_name == 'RCAN':
        model = RCANModel()
        train_dataset = DatasetFromNpyTrain(opt, rgb=True)
        test_dataset = DatasetFromNpyTest(opt, rgb=True)

    elif opt.SR_name == 'EDSR':
        model = EDSRModel()
        train_dataset = DatasetFromNpyTrain(opt, rgb=True)
        test_dataset = DatasetFromNpyTest(opt, rgb=True)

    elif opt.SR_name == 'RDN':
        model = RDNModel()
        train_dataset = DatasetFromNpyTrain(opt, rgb=True)
        test_dataset = DatasetFromNpyTest(opt, rgb=True)
    else:
        raise NotImplementedError('%s is not supported!' % opt.SR_name)

    visualizer = Visualizer()

    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model.initialize(opt)

    if opt.epoch > 0:
        model.load_model(opt.epoch, opt.SR_name)

    for epoch in range(opt.epoch + 1, opt.nEpochs + 1):
        model.set_mode(train=True)

        for i, data in enumerate(train_loader, 1):

            model.set_input(data)
            loss = model.train()
            if i % 50 == 0:
                images = {'input': model.input, 'output': model.output,
                          'target': model.target}
                visualizer.display_current_results(images, k=0)
            if i % 10 == 0:
                print('epoch: {}, iteration: {}/{}, loss: {}'.format(epoch, i, len(train_loader), loss.item()))
                visualizer.plot_current_loss(loss.item())


        model.scheduler.step(epoch)  # update learning rate
        print('Learning rate: %f' % model.scheduler.get_lr()[0])


        if epoch % opt.save_epoch == 0:
            print('a')
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
            log = 'Epoch %d: Average psnr: %f , ssim: %f \n' % (epoch, average_psnr, average_ssim)
            print(log)
            model.log_file.write(log)
            model.save_model(epoch, opt.SR_name)


    model.log_file.close()


if __name__ == '__main__':
    main()