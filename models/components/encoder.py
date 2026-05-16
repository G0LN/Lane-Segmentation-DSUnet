import torch
import torch.nn as nn

class DSConv(nn.Module):
    """Depthwise Separable Convolution Block"""
    def __init__(self, in_channels, out_channels):
        super(DSConv, self).__init__()
        # Depthwise
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Pointwise
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=True, dropout_prob=0.0):
        super(EncoderBlock, self).__init__()
        self.use_pool = use_pool
        # DSUnet uses two DSConv layers per block
        self.conv = nn.Sequential(
            DSConv(in_channels, out_channels),
            DSConv(out_channels, out_channels)
        )
        
        if self.use_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        features = self.conv(x)
        features = self.dropout(features)
        
        if self.use_pool:
            pooled = self.pool(features)
            return features, pooled
        else:
            return features
