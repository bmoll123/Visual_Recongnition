import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNeXt101_32X8D_Weights, resnext101_32x8d
from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d


class EnhancedResNeXt101(nn.Module):
    def __init__(self, num_classes=100, dropout_prob=0.5):
        super(EnhancedResNeXt101, self).__init__()
        base_model = models.resnext101_32x8d(
            weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        )
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(2048, 2048 // 16),
            nn.ReLU(inplace=True),
            nn.Linear(2048 // 16, 2048),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob), nn.Linear(2048, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.channel_attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x_pool = self.avg_pool(x).view(x.size(0), -1)
        att = self.channel_attention(x_pool)
        x_att = x_pool * att
        out = self.classifier(x_att)
        return out


class EnhancedResNeXt50(nn.Module):
    def __init__(self, num_classes=100, dropout_prob=0.5):
        super(EnhancedResNeXt50, self).__init__()
        base_model = models.resnext50_32x4d(
            weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(2048, 2048 // 16),
            nn.ReLU(inplace=True),
            nn.Linear(2048 // 16, 2048),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob), nn.Linear(2048, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.channel_attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x_pool = self.avg_pool(x).view(x.size(0), -1)
        att = self.channel_attention(x_pool)
        x_att = x_pool * att
        out = self.classifier(x_att)
        return out


class ResNeXtBottleneck(nn.Module):

    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        groups=32,
        base_width=4,
    ):
        super(ResNeXtBottleneck, self).__init__()

        width = int(out_channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(
            width, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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


class ResNeXt50_Handcraft(nn.Module):

    def __init__(self, num_classes=100, dropout_prob=0.5, groups=32, width_per_group=4):
        super(ResNeXt50_Handcraft, self).__init__()
        self.in_channels = 64
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResNeXtBottleneck, 64, 3)
        self.layer2 = self._make_layer(ResNeXtBottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResNeXtBottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(ResNeXtBottleneck, 512, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(2048, 2048 // 16),
            nn.ReLU(inplace=True),
            nn.Linear(2048 // 16, 2048),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob), nn.Linear(2048, num_classes)
        )
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                self.groups,
                self.base_width,
            )
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_pool = self.avg_pool(x).view(x.size(0), -1)
        att = self.channel_attention(x_pool)
        out = self.classifier(x_pool * att)
        return out


class ResNeXt101_Handcraft(nn.Module):

    def __init__(self, num_classes=100, dropout_prob=0.5, groups=32, width_per_group=8):
        super(ResNeXt101_Handcraft, self).__init__()
        self.in_channels = 64
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResNeXtBottleneck, 64, 3)
        self.layer2 = self._make_layer(ResNeXtBottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResNeXtBottleneck, 256, 23, stride=2)
        self.layer4 = self._make_layer(ResNeXtBottleneck, 512, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(2048, 2048 // 16),
            nn.ReLU(inplace=True),
            nn.Linear(2048 // 16, 2048),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob), nn.Linear(2048, num_classes)
        )
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                self.groups,
                self.base_width,
            )
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_pool = self.avg_pool(x).view(x.size(0), -1)
        att = self.channel_attention(x_pool)
        out = self.classifier(x_pool * att)
        return out


def resnext50_handcraft(num_classes=100, dropout_prob=0.5, pretrained=True):
    model = ResNeXt50_Handcraft(num_classes=num_classes, dropout_prob=dropout_prob)

    if pretrained:
        base_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        pretrained_dict = base_model.state_dict()
        filtered_dict = {
            k: v for k, v in pretrained_dict.items() if not k.startswith("fc.")
        }

        model_dict = model.state_dict()
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)

    return model


def resnext101_handcraft(num_classes=100, dropout_prob=0.5, pretrained=True):
    model = ResNeXt101_Handcraft(
        num_classes=num_classes, dropout_prob=dropout_prob, groups=32, width_per_group=8
    )

    if pretrained:
        base_model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
        pretrained_dict = base_model.state_dict()
        filtered_dict = {
            k: v for k, v in pretrained_dict.items() if not k.startswith("fc.")
        }

        model_dict = model.state_dict()
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)

    return model
