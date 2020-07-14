import torch
from torch import nn
import torch.nn.functional as F

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

def _SplitChannels(channels, num_groups):
    split_channels = [channels//num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_channels = _SplitChannels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i]//2,
                groups=self.split_channels[i],
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x

# SEffi_A
class BlazeBlock(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,mid_channels=None,stride=1):
        super(BlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride>100000:
            self.use_pool = True
        else:
            self.use_pool = False

        self.mix_branch1 = nn.Sequential(
            MDConv(channels=in_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.mix_branch1(x)
        out = (branch1+self.shortcut(x)) if self.use_pool else branch1
        return self.relu(out)


# SEffi_B
class DoubleBlazeBlock(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,stride=1):
        super(DoubleBlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride > 10000:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            MDConv(channels=in_channels, kernel_size=[3,5], stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.hswish = hswish()

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1 + self.shortcut(x)) if self.use_pool else branch1
        return self.hswish(out)


# SEffiNet
class MixBlazeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MixBlazeNet, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        self.blazeBlock = nn.Sequential(
            BlazeBlock(in_channels=24, out_channels=24, kernel_size=[3,5]),
            BlazeBlock(in_channels=24, out_channels=48, kernel_size=[3,5], stride=2),
        )

        self.doubleBlazeBlock = nn.Sequential(
            DoubleBlazeBlock(in_channels=48, out_channels=96, mid_channels=24, stride=2),
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24, stride=2),
            DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24),
            nn.AvgPool2d(14),
        )
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.blazeBlock(x)
        x = self.doubleBlazeBlock(x)
        x = x.view(-1, 96)
        x = self.fc(x)
        return x


def cal_model():
    from torchstat import stat
    net = MixBlazeNet(num_classes=2)
    stat(net, (3, 224, 224))


if __name__ == "__main__":
    cal_model()
