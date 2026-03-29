import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNeXt101_32X8D_Weights

# ==========================================
# 1. CBAM 模組與 ResNet50_CBAM 實作
# ==========================================

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # 空間注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. 應用通道注意力
        x = x * self.channel_attn(x)
        # 2. 應用空間注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        x = x * self.spatial_attn(spatial_input)
        return x

class CBAMBottleneck(models.resnet.Bottleneck):
    def __init__(self, *args, **kwargs):
        super(CBAMBottleneck, self).__init__(*args, **kwargs)
        self.cbam = CBAM(self.bn3.num_features)

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

        # 在與 Shortcut 相加前執行 CBAM
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

def resnet50_cbam(num_classes=100, pretrained=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for i, b in enumerate(layer):
            stride = b.stride
            downsample = b.downsample
            groups = getattr(b, 'groups', b.conv2.groups)
            dilation = getattr(b, 'dilation', b.conv2.dilation)
            width = b.conv2.out_channels
            
            new_bottleneck = CBAMBottleneck(
                inplanes=b.conv1.in_channels,
                planes=width // groups,
                stride=stride,
                downsample=downsample,
                groups=groups,
                base_width=64,
                dilation=dilation,
                norm_layer=nn.BatchNorm2d
            )
            new_bottleneck.load_state_dict(b.state_dict(), strict=False)
            layer[i] = new_bottleneck
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# ==========================================
# 2. Enhanced ResNeXt101 (SE-style Attention)
# ==========================================

class EnhancedResNeXt101(nn.Module):
    """
    Enhanced ResNeXt101_32X8D with channel attention mechanism.
    """
    def __init__(self, num_classes=100, dropout_prob=0.5):
        super(EnhancedResNeXt101, self).__init__()
        # 載入預訓練的 ResNeXt101_32X8D (IMAGENET1K_V2 是較新的權重)
        base_model = models.resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)

        # 移除最後兩層 (AvgPool 和 FC)
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # 全域平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 通道注意力模組 (Squeeze-and-Excitation 風格)
        # ResNeXt101 最後一層輸出的通道數是 2048
        self.channel_attention = nn.Sequential(
            nn.Linear(2048, 2048 // 16),
            nn.ReLU(inplace=True),
            nn.Linear(2048 // 16, 2048),
            nn.Sigmoid()
        )

        # 分類頭 (包含 Dropout 以解決 Overfitting)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(2048, num_classes)
        )

        # 初始化權重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.channel_attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)                # 提取特徵
        x_pool = self.avg_pool(x).view(x.size(0), -1)  # 池化並展平
        
        # 應用通道注意力
        att = self.channel_attention(x_pool)
        x_att = x_pool * att
        
        # 分類
        out = self.classifier(x_att)
        return out