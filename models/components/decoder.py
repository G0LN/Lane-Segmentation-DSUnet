import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import RepDSConv

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_prob=0.0, deploy=False):
        super(DecoderBlock, self).__init__()
        
        # 1. Asymmetric Decoder: Bilinear Upsampling + Conv 1x1
        # Upsamples spatial dimensions by 2x and projects channels to in_channels // 2
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        )
        
        # 2. Skip Connection Compression: Conv 1x1 to compress skip channels by 4x
        # Helps dramatically reduce memory bandwidth and VRAM overhead
        compressed_skip_channels = max(16, skip_channels // 4)
        self.compress_skip = nn.Conv2d(skip_channels, compressed_skip_channels, kernel_size=1, bias=False)
        self.bn_skip = nn.BatchNorm2d(compressed_skip_channels)
        self.relu_skip = nn.ReLU(inplace=True)
        
        # Total channels after concatenation
        cat_channels = (in_channels // 2) + compressed_skip_channels
        
        # 3. Asymmetric Decoder: Single RepDSConv instead of two DSConv layers
        self.conv = RepDSConv(cat_channels, out_channels, deploy=deploy)
        
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x, skip_features):
        # 1. Bilinear upsampling and channel projection
        x = self.up(x)
        
        # 2. Compress skip features from encoder
        skip_comp = self.compress_skip(skip_features)
        skip_comp = self.bn_skip(skip_comp)
        skip_comp = self.relu_skip(skip_comp)
        
        # Crop or pad skip features if dimensions don't match (due to rounding)
        diffY = skip_comp.size()[2] - x.size()[2]
        diffX = skip_comp.size()[3] - x.size()[3]

        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            
        # Concatenate compressed skip features with upsampled features
        x = torch.cat([skip_comp, x], dim=1)
        
        # Apply single convolution block and dropout
        x = self.conv(x)
        x = self.dropout(x)
        return x

    def switch_to_deploy(self):
        if hasattr(self.conv, 'switch_to_deploy'):
            self.conv.switch_to_deploy()
