import torch.nn as nn
import torch.nn.functional as F
import torch


class PNN(nn.Module):
    def __init__(self, in_channels, kernels, scope):
        super(PNN, self).__init__()

        # Network variables
        self.scope = scope

        # Network structure
        self.conv1 = nn.Conv2d(in_channels, 48, kernels[0])
        self.conv2 = nn.Conv2d(48, 32, kernels[1])
        self.conv3 = nn.Conv2d(32, in_channels - 1, kernels[2])

    def forward(self, inp):
        x = F.relu(self.conv1(inp))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x + inp[:, :-1, self.scope:-self.scope, self.scope:-self.scope]
        return x


class PanNet(nn.Module):
    def __init__(self, nbands, ratio):
        super(PanNet, self).__init__()
        self.ratio = ratio
        self.Conv2d_transpose = nn.ConvTranspose2d(nbands, nbands, 8, 4, padding=(2, 2), bias=False)
        self.Conv = nn.Conv2d(9, 32, 3, padding=(1, 1))
        self.Conv_1 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_3 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_4 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_5 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_6 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_7 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_8 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_9 = nn.Conv2d(32, nbands, 3, padding=(1, 1))

    def forward(self, inp):

        lms = inp[:, :-1, 2::self.ratio, 2::self.ratio]
        pan = torch.unsqueeze(inp[:, -1, :, :], dim=1)

        x = self.Conv2d_transpose(lms)
        x = torch.cat((x, pan), dim=1)

        x1 = F.relu(self.Conv(x))

        x2 = F.relu(self.Conv_1(x1))
        x3 = self.Conv_2(x2) + x1

        x4 = F.relu(self.Conv_3(x3))
        x5 = self.Conv_4(x4) + x3

        x6 = F.relu(self.Conv_5(x5))
        x7 = self.Conv_6(x6) + x5

        x8 = F.relu(self.Conv_7(x7))
        x9 = self.Conv_8(x8) + x7

        x10 = self.Conv_9(x9)

        return x10


class DRPNN(nn.Module):
    def __init__(self, in_channels):
        super(DRPNN, self).__init__()
        self.Conv_1 = nn.Conv2d(in_channels, 64, 7, padding=(3, 3))
        self.Conv_2 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_3 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_4 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_5 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_6 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_7 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_8 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_9 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_10 = nn.Conv2d(64, 9, 7, padding=(3, 3))
        self.Conv_11 = nn.Conv2d(in_channels, in_channels - 1, 3, padding=(1, 1))

    def forward(self, x):
        x1 = F.relu(self.Conv_1(x))
        x2 = F.relu(self.Conv_2(x1))
        x3 = F.relu(self.Conv_3(x2))
        x4 = F.relu(self.Conv_4(x3))
        x5 = F.relu(self.Conv_5(x4))
        x6 = F.relu(self.Conv_6(x5))
        x7 = F.relu(self.Conv_7(x6))
        x8 = F.relu(self.Conv_8(x7))
        x9 = F.relu(self.Conv_9(x8))
        x10 = self.Conv_10(x9)
        x11 = self.Conv_11(F.relu(x10 + x))

        return x11
