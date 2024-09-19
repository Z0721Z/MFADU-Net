import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from my_nets.DAB import AttentionGate, ChannelSpatialSqueezeExcitation


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0)
        )
        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=18, dilation=18)
        self.c5 = Conv2D(out_c * 5, out_c, kernel_size=1, padding=0, dilation=1)

    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear", align_corners=True)

        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)

        xc = torch.cat([x0, x1, x2, x3, x4], axis=1)
        y = self.c5(xc)

        return y


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c, out_c)
        self.a1 = squeeze_excitation_block(out_c)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.a1(x)
        return x


class encoder1(nn.Module):
    def __init__(self):
        super().__init__()

        # self.Resnet50 = resnet50(pretrained=True)
        # print(network)
        network = vgg19(pretrained=True)
        """self.x1 = nn.Sequential(self.Resnet50.conv1, self.Resnet50.bn1, self.Resnet50.relu,
                                nn.Upsample(scale_factor=2))  # 3 64
        self.x2 = nn.Sequential(self.Resnet50.maxpool, *self.Resnet50.layer1)  # 64 256
        self.x3 = nn.Sequential(*self.Resnet50.layer2)  # 256,512
        self.x4 = nn.Sequential(*self.Resnet50.layer3)  # 512 1024
        self.x5 = nn.Sequential(*self.Resnet50.layer4)  # 1024 2048"""
        self.x1 = network.features[:4]
        self.x2 = network.features[4:9]
        self.x3 = network.features[9:18]
        self.x4 = network.features[18:27]
        self.x5 = network.features[27:36]

    def forward(self, x):
        x0 = x
        x1 = self.x1(x0)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)
        return x5, [x4, x3, x2, x1]


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)

        weights = self.conv1(pool)

        weights = self.sigmoid(weights)

        out = x * weights
        return out


class DilatedConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DilatedConvolutionBlock, self).__init__()

        self.dilation_change = nn.Parameter(torch.tensor(dilation, dtype=torch.float32), requires_grad=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, dilation=1)
        self.activation = nn.ReLU()

    def forward(self, x):

        dilation_rate = round(self.dilation_change.item())

        self.conv.dilation = dilation_rate
        self.conv.padding = dilation_rate
        x = self.conv(x)
        x = self.activation(x)
        return x


class DCAM(nn.Module):
    def __init__(self, in_ch=1024 + 1024, out_ch=1024, dilation=2):
        super(DCAM, self).__init__()
        # self.down = nn.Conv2d(in_channels=out_ch * 3, out_channels=out_ch, kernel_size=1, stride=1, padding=0)
        self.up_sample = nn.Sequential(nn.Upsample(scale_factor=2),
                                       nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(out_ch))
        self.SA = SpatialAttention()
        self.fn1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch))
        self.dconv = DilatedConvolutionBlock(in_channels=out_ch, out_channels=out_ch * 2, dilation=dilation)
        self.fn2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch))

        # self.se = squeeze_excitation_block(out_ch // 2)

    def forward(self, d_in, e_in):
        x = torch.cat([d_in, e_in], axis=1)
        x1 = self.fn1(x)
        x1 = self.dconv(x1)
        x1 = self.fn2(x1)
        x1 = self.SA(x1)
        return x1


class Multi_fusion(nn.Module):
    def __init__(self, choose=1, out_channels=64):
        super(Multi_fusion, self).__init__()
        self.choose = choose
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.down_all = nn.Sequential(nn.Conv2d(960, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1), nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1), nn.Sigmoid())
        self.conv3 = nn.Sequential(nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1), nn.Sigmoid())
        self.conv4 = nn.Sequential(nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=1), nn.Sigmoid())
        self.cs1 = ChannelSpatialSqueezeExcitation(64, 2)
        self.cs2 = ChannelSpatialSqueezeExcitation(128, 2)
        self.cs3 = ChannelSpatialSqueezeExcitation(256, 2)
        self.cs4 = ChannelSpatialSqueezeExcitation(512, 2)

    def forward(self, x1, x2, x3, x4):
        x1 = self.cs1(x1)
        x2 = self.cs2(x2)
        x3 = self.cs3(x3)
        x4 = self.cs4(x4)
        if self.choose == 1:
            x2 = self.conv2(self.up(x2))
            for i in range(2):
                x3 = self.up(x3)
                x4 = self.up(x4)
            x3 = self.conv3(x3)
            x4 = self.conv4(self.up(x4))
            x = x1 + x2 + x3 + x4
            return x
        elif self.choose == 2:
            x1 = self.conv1(self.down(x1))
            x3 = self.conv3(self.up(x3))
            for i in range(2):
                x4 = self.up(x4)
            x4 = self.conv4(x4)
            x = x1 + x2 + x3 + x4
            return x
        elif self.choose == 3:
            x2 = self.conv2(self.down(x2))
            for i in range(2):
                x1 = self.down(x1)
            x1 = self.conv1(x1)
            x4 = self.conv4(self.up(x4))
            x = x1 + x2 + x3 + x4
            return x
        elif self.choose == 4:
            for i in range(2):
                x1 = self.down(x1)
                x2 = self.down(x2)
            x1 = self.conv1(self.down(x1))
            x2 = self.conv2(x2)
            x3 = self.conv3(self.down(x3))
            x = x1 + x2 + x3 + x4
            return x


class decoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = DCAM(in_ch=512 + 64, out_ch=512)
        self.c2 = DCAM(in_ch=512 + 256, out_ch=256)
        self.c3 = DCAM(256 + 128, 128)
        self.c4 = DCAM(128 + 64, 64)

        self.mf1 = Multi_fusion(choose=4, out_channels=512)
        self.mf2 = Multi_fusion(choose=3, out_channels=256)
        self.mf3 = Multi_fusion(choose=2, out_channels=128)
        self.mf4 = Multi_fusion(choose=1, out_channels=64)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag1 = AttentionGate(512, 512)
        self.ag2 = AttentionGate(256, 256)
        self.ag3 = AttentionGate(128, 128)
        self.ag4 = AttentionGate(64, 64)

    def forward(self, x, skip):
        s1, s2, s3, s4 = skip

        s1 = self.mf1(s4, s3, s2, s1)
        s2 = self.mf2(s4, s3, s2, s1)
        s3 = self.mf3(s4, s3, s2, s1)
        s4 = self.mf4(s4, s3, s2, s1)

        x = self.up(x)
        x = self.c1(x, s1)
        x = self.ag1(x, s1)

        x = self.up(x)
        x = self.c2(x, s2)
        x = self.ag2(x, s2)

        x = self.up(x)
        x = self.c3(x, s3)
        x = self.ag3(x, s3)

        x = self.up(x)
        x = self.c4(x, s4)
        x = self.ag4(x, s4)

        return x


class encoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.c1 = conv_block(3, 32)
        self.c2 = conv_block(32, 64)
        self.c3 = conv_block(64, 128)
        self.c4 = conv_block(128, 256)

    def forward(self, x):
        x0 = x

        x1 = self.c1(x0)
        p1 = self.pool(x1)

        x2 = self.c2(p1)
        p2 = self.pool(x2)

        x3 = self.c3(p2)
        p3 = self.pool(x3)

        x4 = self.c4(p3)
        p4 = self.pool(x4)

        return p4, [x4, x3, x2, x1]


class decoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(832, 256)
        self.c2 = conv_block(640, 128)
        self.c3 = conv_block(320, 64)
        self.c4 = conv_block(160, 32)

    def forward(self, x, skip1, skip2):
        x = self.up(x)
        x = torch.cat([x, skip1[0], skip2[0]], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, skip1[1], skip2[1]], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, skip1[2], skip2[2]], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, skip1[3], skip2[3]], axis=1)
        x = self.c4(x)

        return x


class build_doubleunet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder1()
        self.a1 = ASPP(512, 64)
        self.d1 = decoder1()
        self.y1 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.e2 = encoder2()
        self.a2 = ASPP(256, 64)
        self.d2 = decoder2()
        self.y2 = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x0 = x
        x, skip1 = self.e1(x)
        x = self.a1(x)
        x = self.d1(x, skip1)
        y1 = self.y1(x)

        input_x = x0 * self.sigmoid(y1)
        x, skip2 = self.e2(input_x)
        x = self.a2(x)
        x = self.d2(x, skip1, skip2)
        y2 = self.y2(x)

        return y2


if __name__ == "__main__":
    x = torch.randn((4, 3, 128, 128))
    model = build_doubleunet()
    y2 = model(x)
    print(y2.shape)
