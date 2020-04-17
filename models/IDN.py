import torch
import torch.nn as nn
from .base_model import BaseModel, init_weights
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import ycbcr2rgb, quantize
# DBlocks
class Enhancement_unit(nn.Module):
    def __init__(self, nFeat, nDiff, nFeat_slice):
        super(Enhancement_unit, self).__init__()

        self.D3 = nFeat
        self.d = nDiff
        self.s = nFeat_slice

        block_0 = []
        block_0.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=False))
        block_0.append(nn.LeakyReLU(0.05))

        # Group Convolution
        block_0.append(nn.Conv2d(nFeat - nDiff, nFeat - 2 * nDiff, kernel_size=3, padding=1, bias=False))
        # block_0.append(SeparableConv2d(nFeat-nDiff, nFeat-2*nDiff, kernel_size=3, padding=1, bias=False))

        block_0.append(nn.LeakyReLU(0.05))
        block_0.append(nn.Conv2d(nFeat-2*nDiff, nFeat, kernel_size=3, padding=1, bias=False))
        block_0.append(nn.LeakyReLU(0.05))
        self.conv_block0 = nn.Sequential(*block_0)

        block_1 = []

        # Group Convolution
        block_1.append(nn.Conv2d(nFeat - nFeat // 4, nFeat, kernel_size=3, padding=1, bias=False))
        # block_0.append(SeparableConv2d(nFeat - nFeat // 4, nFeat, kernel_size=3, padding=1, bias=False))

        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat, nFeat-nDiff, kernel_size=3, padding=1, bias=False))
        block_1.append(nn.LeakyReLU(0.05))
        block_1.append(nn.Conv2d(nFeat-nDiff, nFeat+nDiff, kernel_size=3, padding=1, bias=False))
        block_1.append(nn.LeakyReLU(0.05))
        self.conv_block1 = nn.Sequential(*block_1)
        self.compress = nn.Conv2d(nFeat+nDiff, nFeat, kernel_size=1, padding=0, bias=False)

    def forward(self, x):

        x_feature_shot = self.conv_block0(x)
        feature = x_feature_shot[:,0:(self.D3-self.D3//self.s),:,:]
        feature_slice = x_feature_shot[:,(self.D3-self.D3//self.s):self.D3,:,:]
        x_feat_long = self.conv_block1(feature)
        feature_concat = torch.cat((feature_slice, x), 1)
        out = x_feat_long + feature_concat
        out = self.compress(out)
        return out


class IDN(nn.Module):
    def __init__(self, Scale, in_channel, out_channel):
        super(IDN, self).__init__()
        nFeat = 64
        nDiff = 16
        nFeat_slice = 4
        scale = Scale
        self.scale = Scale
        self.conv1 = nn.Conv2d(in_channel, nFeat, kernel_size=3, padding=1, bias=False)
        self.lrelu1 = nn.LeakyReLU(0.05)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False)
        self.lrelu2 = nn.LeakyReLU(0.05)
        self.Enhan_unit1 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit2 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit3 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        self.Enhan_unit4 = Enhancement_unit(nFeat, nDiff, nFeat_slice)
        # Upsampler
        # self.upsample = nn.ConvTranspose2d(nFeat, nChannel, stride=self.scale, kernel_size=17, padding=8)
        if self.scale == 2 or self.scale == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(nFeat, nFeat * scale * scale, kernel_size=3, padding=1, bias=True),
                sub_pixel(scale)
            ])
        elif self.scale == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(nFeat, nFeat * scale, kernel_size=3, padding=1, bias=True),
                sub_pixel(scale // 2),
                nn.Conv2d(nFeat, nFeat * scale, kernel_size=3, padding=1, bias=True),
                sub_pixel(scale // 2)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")
        # conv
        self.conv3 = nn.Conv2d(nFeat, out_channel, kernel_size=3, padding=1, bias=True)

    def forward(self, x, x_bicubic):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))

        x = self.Enhan_unit1(x)
        x = self.Enhan_unit2(x)
        x = self.Enhan_unit3(x)
        x = self.Enhan_unit4(x)

        # x_upsample = self.upsample(x, output_size=x_bicubic.size())
        x_upsample = self.UPNet(x)

        out = self.conv3(x_upsample) + x_bicubic

        return out

class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x


class IDNModel(BaseModel):
    def initialize(self, opt):

        BaseModel.initialize(self, opt)

        self.model = IDN(opt.sr_factor, 1, 1)
        self.criterion = nn.MSELoss()
        self.model.to(self.device)
        self.model = nn.DataParallel(self.model, opt.gpu_ids)
        self.criterion = self.criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        # init_weights(self.model, 'xavier')

    def eval_initialize(self, opt):

        BaseModel.eval_initialize(self, opt)

        self.model = IDN(opt.sr_factor, 1, 1).to(self.device)
        self.model = nn.DataParallel(self.model, opt.gpu_ids)

    def set_input(self, input):
        self.input = input['input'].to(self.device)
        self.target = input['target'].to(self.device)
        self.input_bicu = input['input_up'].to(self.device)


    def train(self):
        self.output = self.model(self.input, self.input_bicu)

        loss = self.criterion(self.output, self.target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.output = quantize(self.output, self.opt.rgb_range)
        return loss

    def set_eval_input(self, input):
        self.eval_input = input['input'].to(self.device)
        self.eval_target = input['target'].to(self.device)
        self.eval_input_bicu = input['input_up'].to(self.device)
        self.eval_input_y = self.eval_input[:, 0, :, :].unsqueeze(1)
        self.eval_target_y = self.eval_target[:, 0, :, :].unsqueeze(1)
        self.eval_input_bicu_y = self.eval_input_bicu[:, 0, :, :].unsqueeze(1)


    def eval(self):
        with torch.no_grad():
            output = self.model(self.eval_input_y, self.eval_input_bicu_y)
            output = quantize(output, self.opt.rgb_range)

        self.output = self.eval_input_bicu.data.clone()
        self.output[:, 0, :, :] = output
        self.output = self.output[0].cpu().permute(1, 2, 0).numpy()
        self.output = ycbcr2rgb(self.output)
        self.output = torch.from_numpy(self.output).permute(2, 0, 1)

        self.eval_input = self.eval_input[0].cpu().permute(1, 2, 0).numpy()
        self.eval_input = ycbcr2rgb(self.eval_input)
        self.eval_input = torch.from_numpy(self.eval_input).permute(2, 0, 1)

        self.eval_target = self.eval_target[0].cpu().permute(1, 2, 0).numpy()
        self.eval_target = ycbcr2rgb(self.eval_target)
        self.eval_target = torch.from_numpy(self.eval_target).permute(2, 0, 1)

        return {'input': self.eval_input, 'output': self.output,
                'target': self.eval_target}