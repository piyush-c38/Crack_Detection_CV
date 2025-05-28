import torch
import torch.nn as nn
import torchvision.models as models

class PSPNet(nn.Module):
    def __init__(self, num_classes=1):
        super(PSPNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.ppm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        )

        self.final = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        return torch.sigmoid(x)