


import torch
import torch.nn as nn
from .base import ObjDetectionModule
from .layerops import ModuleList, Sequential


def conv3x3(in_channels, out_channels, stride=1, padding=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)


class Convolutional(ObjDetectionModule):

    def __init__(
        self, in_channels: int, out_channels: int, 
        kernel_size: int, stride: int
    ) -> None:
        super().__init__()

        if kernel_size == 3:
            self.conv = conv3x3(in_channels, out_channels, stride)
        elif kernel_size == 1:
            self.conv = conv1x1(in_channels, out_channels, stride)
        else:
            raise ValueError(f"kernel size should be 1 or 3 but {kernel_size} received ...")
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class DarkBlock(ObjDetectionModule):

    def __init__(self, num_filters: int, repeats: int, activation: str = 'leaky') -> None:
        super().__init__()

        self.head = Convolutional(num_filters // 2, num_filters, 3, 2)
        self.inner = ModuleList()
        for _ in range(repeats):
            block = Sequential(
                Convolutional(num_filters, num_filters // 2, 1, 1),
                Convolutional(num_filters // 2, num_filters, 3, 1)
            )
            self.inner.append(block)
        
    def forward(self, inputs):
        inputs = self.head(inputs)
        for block in self.inner:
            outs = block(inputs) + inputs
            inputs = outs
        return outs


class ConvolutionalSet(ObjDetectionModule):

    def __init__(self, in_channels: int, num_filters: int) -> None:
        super().__init__()
        self.inner = ModuleList()
        num_filters = [num_filters, num_filters * 2, num_filters, num_filters * 2, num_filters]
        kernel_sizes = [1, 3, 1, 3, 1]
        for num_filter, kernel_size in zip(num_filters, kernel_sizes):
            self.inner.append(Convolutional(in_channels, num_filter, kernel_size, 1))
            in_channels = num_filter

    def forward(self, x):
        for block in self.inner:
            x = block(x)
        return x


class RouteLayer(ObjDetectionModule):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.inner = Sequential(
            Convolutional(in_channels, out_channels, 1, 1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        return self.inner(x)


class YoLoLayer(ObjDetectionModule):
    CLUSTERS = { # (width, height)
        '1': ((10, 13), (16, 30), (33, 23)),
        '2': ((30, 61), (62, 45), (59, 119)),
        '3': ((116, 90), (156, 198), (373, 326))
    }
    def __init__(
        self, in_channels: int, out_channels: int, num_classes: int, 
        cluster: str, stride: int = 32
    ) -> None:
        super().__init__()

        self.stride = stride
        self.anchors = torch.tensor(self.CLUSTERS[cluster], dtype=torch.float32)

        final_channels = self.original_anchors.size(1) * (num_classes + 5)
        self.inner = nn.Sequential(
            Convolutional(in_channels, out_channels, 3, 1),
            conv1x1(out_channels, final_channels, bias=True)
        )
    
    def forward(self, x):
        t = self.inner(x)
        B, _, H, W = t.size()
        t = t.view(B, self.anchors.size(0), -1, H, W).permute((0, 3, 4, 1, 2))
        X, Y = torch.meshgrid(torch.arange(H), torch.arange(W))
        grids = torch.stack((X, Y), 2).view(1, H, W, 1, 2).float().to(x.device)
        anchors = self.anchors().to(x.device).clone().view(1, 1, 1, -1, 2)
        t[..., 0:2] = (t[..., 0:2].sigmoid() + grids) * self.stride
        t[..., 2:4] = t[..., 2:4].exp() * anchors
        if not self.training:
            t[..., 4:].sigmoid_()
        return t


class DarkNet(ObjDetectionModule):

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.head = Convolutional(3, 32, 3, 1) # 32 x 480 x 480
        self.block1 = DarkBlock(64, 1) # 64 x 240 x 240
        self.block2 = DarkBlock(128, 2) # 128 x 120 x 120
        self.block3 = DarkBlock(256, 8) # 256 x 60 x 60
        self.block4 = DarkBlock(512, 8) # 512 x 30 x 30
        self.block5 = DarkBlock(1024, 4) # 1024 x 15 x 15

        self.set1 = ConvolutionalSet(1024, 512)
        self.yolo1 = YoLoLayer(512, 1024, num_classes, cluster='3', stride=32)
        self.route1 = RouteLayer(512, 256)

        self.set2 = ConvolutionalSet(768, 256)
        self.yolo2 = YoLoLayer(256, 512, num_classes, cluster='2', stride=16)
        self.route2 = RouteLayer(256, 128)

        self.set3 = ConvolutionalSet(384, 128)
        self.yolo3 = YoLoLayer(128, 256, num_classes, cluster='1', stride=8)

    def forward(self, x):
        x = self.head(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        results = []
        x = self.set1(x5)
        out1 = self.yolo1(x)
        x = torch.cat((self.route1(x), x4), dim=1)
        results.append([out1, self.yolo1])

        x = self.set2(x)
        out2 = self.yolo2(x)
        x = torch.cat((self.route2(x), x3), dim=1)
        results.append([out2, self.yolo2])

        x = self.set3(x)
        out3 = self.yolo3(x)
        results.append([out3, self.yolo3])

        return results


if __name__ == "__main__":

    darknet = DarkNet(80)
    dummy = torch.rand(2, 3, 416, 416)
    darknet(dummy)
    
