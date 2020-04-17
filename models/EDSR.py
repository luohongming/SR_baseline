

from models import common

import torch.nn as nn
from models.base_model import BaseModel
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import quantize

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        n_resblock = args['n_resblocks']
        n_feats = args['n_feats']
        kernel_size = 3
        scale = args['scale']
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args['rgb_range'], rgb_mean, rgb_std)

        # define head module
        m_head = [conv(args['n_colors'], n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args['res_scale']) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args['n_colors'], kernel_size)
        ]

        self.add_mean = common.MeanShift(args['rgb_range'], rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {},'
                                           'whose dimensions in the model are {} and'
                                           'whose dimensions in the checkpoint are {}.'.format(
                            name, own_state[name].size(), param.size()))

            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))



class EDSRModel(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        args = {'scale': opt.sr_factor, 'n_feats': 256, 'n_resblocks': 32, 'res_scale': 0.1, 'n_colors': 3,
                'rgb_range': opt.rgb_range}
        self.model = EDSR(args).to(self.device)
        self.criterion = nn.L1Loss()
        self.model = nn.DataParallel(self.model, opt.gpu_ids)
        # self.criterion = self.criterion.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)

    def eval_initialize(self, opt):
        BaseModel.eval_initialize(self, opt)

        args = {'scale': opt.sr_factor, 'n_feats': 256, 'n_resblocks': 32, 'res_scale': 0.1, 'n_colors': 3,
                'rgb_range': opt.rgb_range}
        self.model = EDSR(args).to(self.device)
        self.model = nn.DataParallel(self.model, opt.gpu_ids)


    def set_input(self, input):
        self.input = input['input'].to(self.device)
        self.target = input['target'].to(self.device)

    def train(self):
        self.output = self.model(self.input)
        loss = self.criterion(self.output, self.target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.output = quantize(self.output, self.opt.rgb_range)

        return loss

    def set_eval_input(self, input):
        self.eval_input = input['input'].to(self.device)
        self.eval_target = input['target'].to(self.device)


    def eval(self):
        with torch.no_grad():
            output = self.model(self.eval_input)
            output = quantize(output, self.opt.rgb_range)

        return {'input': self.eval_input[0], 'output': output[0], 'target': self.eval_target[0]}