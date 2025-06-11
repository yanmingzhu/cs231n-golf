import torch
import torch.nn as nn
from torch.autograd import Variable
from MobileNetV2 import MobileNetV2
from torchvision import models
from golfdb.util import freeze_layers

class VITLSTM(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(VITLSTM, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        net = models.vit_b_16(pretrained=pretrain)
        net.heads = nn.Identity()

        self.vit = net
        train_layer = []
        for name, param in self.vit.named_parameters():
            if not any([layer in name for layer in train_layer]):
                param.requires_grad = False

        self.rnn = nn.LSTM(int(768*width_mult if width_mult > 1.0 else 768),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.lin = nn.Linear(2*self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))
        else:
            return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.vit(c_in)
        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size*timesteps,9)

        return out
