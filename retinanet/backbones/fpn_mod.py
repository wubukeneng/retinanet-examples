import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet as vrn

from .resnet import ResNet
from .utils import register

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes * (base_width / 64.)) * groups
        
        downsample = nn.Sequential(
            vrn.conv1x1(inplanes, planes * self.expansion, stride),
            norm_layer(planes * self.expansion)
        )
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = vrn.conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = vrn.conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = vrn.conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class FPN_Mod(nn.Module):
    'Feature Pyramid Network - https://arxiv.org/abs/1612.03144'

    def __init__(self, features):
        super().__init__()

        self.stride = 128
        self.features = features

        is_light = features.bottleneck == vrn.BasicBlock
        channels = [128, 256, 512] if is_light else [512, 1024, 2048]

        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        self.lateral6 = nn.Conv2d(channels[2], 256, 1)
        self.lateral7 = nn.Conv2d(256, 256, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth6 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth7 = nn.Conv2d(256, 256, 3, padding=1)
        
        # add c6 and c7 for better large predictions
        self.feature6 = Bottleneck(
            channels[2],
            channels[2],
            stride=2,
            norm_layer=features._norm_layer
        )
        self.feature7 = Bottleneck(
            channels[2],
            256,
            stride=2,
            norm_layer=features._norm_layer
        )

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        self.apply(init_layer)

        self.features.initialize()

    def forward(self, x):
        c3, c4, c5 = self.features(x)
        c6 = self.feature6(c5)
        c7 = self.feature7(c6)
        
        p7 = self.lateral7(c7)
        p6 = self.lateral6(c6)
        p6 = F.interpolate(p7, scale_factor=2) + p6
        p5 = self.lateral5(c5)
        p5 = F.interpolate(p6, scale_factor=2) + p5
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, scale_factor=2) + p4
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, scale_factor=2) + p3

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)
        p6 = self.smooth5(p6)
        p7 = self.smooth5(p7)

        return [p3, p4, p5, p6, p7]
    
@register
def ResNet18FPN_Mod():
    return FPN_Mod(ResNet(layers=[2, 2, 2, 2], bottleneck=vrn.BasicBlock, outputs=[3, 4, 5], url=vrn.model_urls['resnet18']))

@register
def ResNet34FPN_Mod():
    return FPN_Mod(ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.BasicBlock, outputs=[3, 4, 5], url=vrn.model_urls['resnet34']))

@register
def ResNet50FPN_Mod():
    return FPN_Mod(ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, outputs=[3, 4, 5], url=vrn.model_urls['resnet50']))

@register
def ResNet101FPN_Mod():
    return FPN_Mod(ResNet(layers=[3, 4, 23, 3], bottleneck=vrn.Bottleneck, outputs=[3, 4, 5], url=vrn.model_urls['resnet101']))

@register
def ResNet152FPN_Mod():
    return FPN_Mod(ResNet(layers=[3, 8, 36, 3], bottleneck=vrn.Bottleneck, outputs=[3, 4, 5], url=vrn.model_urls['resnet152']))

@register
def ResNeXt50_32x4dFPN_Mod():
    return FPN_Mod(ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, outputs=[3, 4, 5], groups=32, width_per_group=4, url=vrn.model_urls['resnext50_32x4d']))

@register
def ResNeXt101_32x8dFPN_Mod():
    return FPN_Mod(ResNet(layers=[3, 4, 23, 3], bottleneck=vrn.Bottleneck, outputs=[3, 4, 5], groups=32, width_per_group=8, url=vrn.model_urls['resnext101_32x8d']))
