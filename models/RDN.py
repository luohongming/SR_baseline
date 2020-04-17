
from models import common
import torch
import torch.nn as nn
from models.base_model import BaseModel
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import quantize

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args['scale']
        G0 = args['G0']
        kSize = args['RDNkSize']

        #number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args['RDNconfig']]

        #Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args['n_colors'], G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        #Residual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        #Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        #Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args['n_colors'], kSize, padding=(kSize-1)//2, stride=1)
            ])

        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args['n_colors'], kSize, padding=(kSize-1)//2, stride=1)
            ])

        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        return self.UPNet(x)


class RDNModel(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        args = {'scale': opt.sr_factor, 'G0': 64, 'RDNkSize': 3, 'RDNconfig': 'B', 'n_colors': 3}

        self.model = RDN(args).to(self.device)
        self.criterion = nn.MSELoss()
        self.model = nn.DataParallel(self.model, opt.gpu_ids)
        # self.criterion = self.criterion.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)


    def eval_initialize(self, opt):
        BaseModel.eval_initialize(self, opt)

        args = {'scale': opt.sr_factor, 'G0': 64, 'RDNkSize': 3, 'RDNconfig': 'B', 'n_colors': 3}
        self.model = RDN(args).to(self.device)
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