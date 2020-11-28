# coding: utf-8
import math
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable


class StandardCNN(nn.Module):
    def __init__(self, inputDim=512, hiddenDim=256):
        super(StandardCNN, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        # frontend1D
        self.fronted1D = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.Conv1d(64, 64, kernel_size=20, stride=4, padding=8, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.AvgPool1d(kernel_size=5, padding=0)
                )
        # BiGRU
        self.bigru1 = nn.GRU(self.inputDim, self.hiddenDim, bidirectional=True, batch_first=True)
        self.bigru2 = nn.GRU(2 * self.hiddenDim, self.hiddenDim, bidirectional=True, batch_first=True)
        # Linear
        self.linear = nn.Linear(2 * self.hiddenDim, 28) # 26 + space + blank

        # initialize
        self._initialize_weights()

    def load_ckpt(self,ckpt="./WER/model_best_16k.pth.tar"):
        pretrained_dict = torch.load(ckpt)
        self.load_state_dict(pretrained_dict['state_dict'])

    def forward(self, x, length=None):
        x = x * 32768
        frameLen = x.size(2)
        x = self.fronted1D(x)
        length = [_//640 for _ in length]
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, frameLen//640, 512)
        x = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True)
        x, _ = self.bigru1(x)
        x, _ = self.bigru2(x)
        x, l = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.linear(x)
        # -- output Time x Batch x Alphabet_size
        x = x.transpose(0, 1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
