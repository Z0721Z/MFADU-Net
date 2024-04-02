import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', activation=F.relu):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.activation = activation

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.activation(x)


class InceptionSepConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionSepConvBlock, self).__init__()
        self.conva = nn.Sequential(
            SeparableConv2d(in_channels, in_channels, (3, 3), padding='same'),
            SeparableConv2d(in_channels, in_channels//2, (3, 3), padding='same')
        )

        self.convb = nn.Sequential(
            SeparableConv2d(in_channels, in_channels, (5, 5), padding='same'),
            SeparableConv2d(in_channels, in_channels//2, (5, 5), padding='same')
        )

        self.residual = nn.Conv2d(in_channels, in_channels, 1, padding='same')

    def forward(self, x):
        conva = self.conva(x)
        convb = self.convb(x)

        concatenated = torch.cat([conva, convb], 1)
        residual = self.residual(x)

        return F.relu(concatenated + residual)


class ChannelSqueezeExcitation(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2, activation='relu'):
        super(ChannelSqueezeExcitation, self).__init__()
        self.num_channels = num_channels
        self.reduction_ratio = nn.Parameter(torch.tensor(reduction_ratio, dtype=torch.float32), requires_grad=True)

        # Convert self.reduction_ratio to an integer
        reduction_channels = int(num_channels // self.reduction_ratio.item())

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_channels, reduction_channels)
        self.fc2 = nn.Linear(reduction_channels, num_channels)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global_avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        fc1 = self.activation(self.fc1(global_avg_pool))
        fc2 = self.sigmoid(self.fc2(fc1)).view(x.size(0), -1, 1, 1)
        output_tensor = x * fc2
        return output_tensor


class SpatialSqueezeExcitation(nn.Module):
    def __init__(self, in_channels):
        super(SpatialSqueezeExcitation, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        squeeze_tensor = self.global_avg_pool(x)
        squeeze_tensor = self.sigmoid(self.conv(squeeze_tensor))
        output_tensor = x * squeeze_tensor
        return output_tensor


class ChannelSpatialSqueezeExcitation(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSqueezeExcitation, self).__init__()
        self.channel_se = ChannelSqueezeExcitation(num_channels, reduction_ratio)
        self.spatial_se = SpatialSqueezeExcitation(num_channels)

    def forward(self, x):
        cse_output = self.channel_se(x)
        sse_output = self.spatial_se(x)
        output_tensor = cse_output + sse_output
        return output_tensor


class DAB(nn.Module):
    def __init__(self, num_channels):
        super(DAB, self).__init__()
        self.deepconv = InceptionSepConvBlock(num_channels)
        self.CSA = ChannelSpatialSqueezeExcitation(num_channels)
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.deepconv(x)
        x = self.CSA(x)
        return self.down(x)


class AttentionGate(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(AttentionGate, self).__init__()

        self.gate_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.inputs_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.psi_conv = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, gate):
        g1 = self.gate_conv(gate)
        x1 = self.inputs_conv(inputs)
        psi = self.leaky_relu(g1 + x1)
        psi = self.psi_conv(psi)
        psi = self.sigmoid(psi)
        out = inputs * psi
        return out


if __name__ == '__main__':
    model = DAB(64)
    model2 = AttentionGate(64, 32)
    input = torch.randn(4, 64, 128, 128)
    out = model(input)
    # out = model2(out, input)
    print(out.shape)
