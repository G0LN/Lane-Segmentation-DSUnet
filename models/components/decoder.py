import torch
import torch.nn as nn
from .encoder import DSConv

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_prob=0.0):
        super(DecoderBlock, self).__init__()
        # Up-Conv 2x2
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # DSConv after concatenation
        self.conv = nn.Sequential(
            DSConv(in_channels // 2 + skip_channels, out_channels),
            DSConv(out_channels, out_channels)
        )
        
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x, skip_features):
        x = self.up(x)
        
        # Crop skip_features if dimensions don't match (due to arbitrary padding/sizing)
        # B, C, H, W
        diffY = skip_features.size()[2] - x.size()[2]
        diffX = skip_features.size()[3] - x.size()[3]

        if diffY > 0 or diffX > 0:
            import torch.nn.functional as F
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            
        # Concatenate
        x = torch.cat([skip_features, x], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x
