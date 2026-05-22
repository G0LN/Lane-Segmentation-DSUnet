import torch
import torch.nn as nn
from .components.encoder import EncoderBlock, DSConv
from .components.decoder import DecoderBlock

class DSUnet(nn.Module):
    """
    DSUnet (Dual Stream Unet) Architecture for Lane Detection / Segmentation.
    Using Depthwise Separable Convolutions to reduce parameters.
    """
    def __init__(self, in_channels=3, num_classes=4, dropout=0.5, width_multiplier=1.0):
        super(DSUnet, self).__init__()
        
        # Scale number of channels dynamically based on width_multiplier
        c1 = int(64 * width_multiplier)
        c2 = int(128 * width_multiplier)
        c3 = int(256 * width_multiplier)
        c4 = int(512 * width_multiplier)
        c5 = int(1024 * width_multiplier) # Bottleneck
        
        # As per DSUNet typical structure:
        # Dropout layers are added in the deeper layers, now configurable via a single parameter
        
        self.enc1 = EncoderBlock(in_channels, c1, use_pool=True)
        self.enc2 = EncoderBlock(c1, c2, use_pool=True)
        self.enc3 = EncoderBlock(c2, c3, use_pool=True)
        self.enc4 = EncoderBlock(c3, c4, use_pool=True, dropout_prob=dropout)
        
        # Bottleneck (No pooling)
        self.bottleneck = EncoderBlock(c4, c5, use_pool=False, dropout_prob=dropout)
        
        self.dec4 = DecoderBlock(c5, c4, c4, dropout_prob=dropout) # The third dropout layer
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        
        # Final prediction layer: 1x1 Standard Conv
        # We output logits (raw values). If binary, train with BCEWithLogitsLoss.
        # If multiclass, train with CrossEntropyLoss.
        self.out_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoding
        skip1, p1 = self.enc1(x)
        skip2, p2 = self.enc2(p1)
        skip3, p3 = self.enc3(p2)
        skip4, p4 = self.enc4(p3)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoding
        d4 = self.dec4(b, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        
        out = self.out_conv(d1)
        return out

if __name__ == "__main__":
    # Test model shape and print parameters
    model = DSUnet(in_channels=3, num_classes=4)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")
