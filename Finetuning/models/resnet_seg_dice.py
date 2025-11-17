import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2) 
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,#input
            64,#output
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        

        self.upconv3 = nn.ConvTranspose3d(512 * block.expansion, 512, kernel_size=2, stride=1,padding=0)
        self.decoder_conv3 = DecoderBlock(512 + 1024, 512)
        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=1, padding=0)
        self.decoder_conv2 = DecoderBlock(256 + 512, 256)
        self.upconv1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv1 =DecoderBlock(128 + 256, 128)
        self.upconv0 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=1,padding=0)
        self.decoder_conv0 = DecoderBlock(64+64,64)
        self.final_conv = nn.Conv3d(64, num_seg_classes, kernel_size=1)

        
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _crop_and_concat(self, x1, x2):
        diffD = x1.size()[2] - x2.size()[2]
        diffH = x1.size()[3] - x2.size()[3]
        diffW = x1.size()[4] - x2.size()[4]
        if diffD != 0 or diffH != 0 or diffW != 0:
            x1 = x1[:, :, diffD // 2:-(diffD - diffD // 2), diffH // 2:-(diffH - diffH // 2), diffW // 2:-(diffW - diffW // 2)]
        return torch.cat([x1, x2], dim=1)
    
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Decoder
        d3 = self.upconv3(x4)
        d3 = F.interpolate(d3, size=x3.size()[2:], mode='trilinear', align_corners=True)
        d3 = self._crop_and_concat(d3, x3)
        d3 = self.decoder_conv3(d3)

        d2 = self.upconv2(d3)
        d2 = F.interpolate(d2, size=x2.size()[2:], mode='trilinear', align_corners=True)
        d2 = self._crop_and_concat(d2, x2)
        d2 = self.decoder_conv2(d2)

        d1 = self.upconv1(d2)
        d1 = F.interpolate(d1, size=x1.size()[2:], mode='trilinear', align_corners=True)
        d1 = self._crop_and_concat(d1, x1)
        d1 = self.decoder_conv1(d1)

        d0 = self.upconv0(d1)
        d0 = F.interpolate(d0, size=x0.size()[2:], mode='trilinear', align_corners=True)
        d0 = self._crop_and_concat(d0, x0)
        d0 = self.decoder_conv0(d0)

        out = self.final_conv(d0)
        out = F.interpolate(out, size=(80, 128, 80), mode='trilinear', align_corners=False)
        

        return out

class ResNet_Seg(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet_Seg, self).__init__()
        self.conv1 = nn.Conv3d(
            1,#input
            64,#output
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        # Decoder
        self.decoder = nn.ModuleDict({
            'up4': nn.Sequential(
                nn.ConvTranspose3d(2048, 1024, kernel_size=3, stride=1, padding=1),
                DecoderBlock(2048, 1024)  
            ),

            'up3': nn.Sequential(
                nn.ConvTranspose3d(1024, 512, kernel_size=3, stride=1, padding=1),
                DecoderBlock(1024, 512) 
            ),

            'up2': nn.Sequential(
                nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
                DecoderBlock(512, 256) 
            ),

            'up1': nn.Sequential(
                nn.ConvTranspose3d(256, 64, kernel_size=3, stride=1, padding=1),
                DecoderBlock(128, 64)  
            ),

            'final': nn.Sequential(
                nn.ConvTranspose3d(64, 64, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm3d(64),
                nn.ConvTranspose3d(64, num_seg_classes, kernel_size=1, stride=1)
            )
        })

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x0 = self.conv1(x)    
        x0 = self.bn(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)  
        x2 = self.layer2(x1)  
        x3 = self.layer3(x2)   
        x4 = self.layer4(x3)   

        # Decoder
        d4 = self.decoder['up4'][0](x4) 
        d4 = torch.cat([d4, x3], dim=1) 
        d4 = self.decoder['up4'][1](d4)  

        d3 = self.decoder['up3'][0](d4)  
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.decoder['up3'][1](d3)  
        

        d2 = self.decoder['up2'][0](d3)  
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.decoder['up2'][1](d2)  

        d1 = self.decoder['up1'][0](d2)  
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.decoder['up1'][1](d1)  

        out = self.decoder['final'](d1)  

        return out
    
def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet_Seg(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet_Seg(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet_Seg(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet_Seg(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet_Seg(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet_Seg(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet_Seg(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
