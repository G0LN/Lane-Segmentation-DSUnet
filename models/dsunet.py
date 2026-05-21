import torch
import torch.nn as nn
from .components.encoder import EncoderBlock, DSConv
from .components.decoder import DecoderBlock

class DSUnet(nn.Module):
    """
    DSUnet (Dual Stream Unet) Architecture for Lane Detection / Segmentation.
    Using Depthwise Separable Convolutions to reduce parameters.
    """
    def __init__(self, in_channels=3, num_classes=4, dropout=0.5):
        super(DSUnet, self).__init__()
        
        # As per DSUNet typical structure:
        # Dropout layers are added in the deeper layers, now configurable via a single parameter
        
        self.enc1 = EncoderBlock(in_channels, 64, use_pool=True)
        self.enc2 = EncoderBlock(64, 128, use_pool=True)
        self.enc3 = EncoderBlock(128, 256, use_pool=True)
        self.enc4 = EncoderBlock(256, 512, use_pool=True, dropout_prob=dropout)
        
        # Bottleneck (No pooling)
        self.bottleneck = EncoderBlock(512, 1024, use_pool=False, dropout_prob=dropout)
        
        self.dec4 = DecoderBlock(1024, 512, 512, dropout_prob=dropout) # The third dropout layer
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        
        # Final prediction layer: 1x1 Standard Conv
        # We output logits (raw values). If binary, train with BCEWithLogitsLoss.
        # If multiclass, train with CrossEntropyLoss.
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

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
