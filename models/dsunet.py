import torch
import torch.nn as nn
from .components.encoder import EncoderBlock
from .components.decoder import DecoderBlock

class DSUnet(nn.Module):
    """
    Bilateral Asymmetric DSUnet (B-DSUnet Asymmetric D-Stem4) Architecture.
    Optimized with a Stem-4 Block for 4x early downsampling (saving massive FLOPs),
    an Asymmetric Channel Scaling design ([16, 32, 160, 320, 640] channels at alpha=0.5),
    a Direct Spatial Detail Injection Branch to preserve high-res boundary details at 256x512,
    an Asymmetric Decoder, and Skip Connection Compression.
    """
    def __init__(self, in_channels=3, num_classes=4, dropout=0.5, width_multiplier=1.0, deploy=False):
        super(DSUnet, self).__init__()
        self.deploy = deploy
        
        # Base channels: [16, 32, 160, 320, 640] at width_multiplier=0.5 (Option 1)
        scale = width_multiplier / 0.5
        c1 = max(16, int(16 * scale))
        c2 = max(32, int(32 * scale))
        c3 = max(64, int(160 * scale))
        c4 = max(128, int(320 * scale))
        c5 = max(256, int(640 * scale)) # Bottleneck
        
        # 1. Stem Block (Stride-4 early downsampling):
        # We split it into two consecutive stride-2 Conv blocks to allow capturing
        # an intermediate skip connection at 128x256 resolution.
        self.stem1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Direct Spatial Detail Injection Branch:
        # Projects raw high-res image (256x512) directly to c1 channels to serve as
        # skip connection detail for the final decoder block dec1, bypassing the entire Encoder.
        self.spatial_projector = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # 3. Encoder Path (Starts at 64x128 resolution)
        self.enc2 = EncoderBlock(c1, c2, use_pool=True, deploy=deploy)
        self.enc3 = EncoderBlock(c2, c3, use_pool=True, deploy=deploy)
        self.enc4 = EncoderBlock(c3, c4, use_pool=True, dropout_prob=dropout, deploy=deploy)
        
        # Bottleneck (No pooling, 8x16 resolution)
        self.bottleneck = EncoderBlock(c4, c5, use_pool=False, dropout_prob=dropout, deploy=deploy)
        
        # 4. Decoder Path
        self.dec4 = DecoderBlock(c5, c4, c4, dropout_prob=dropout, deploy=deploy)
        self.dec3 = DecoderBlock(c4, c3, c3, deploy=deploy)
        self.dec2 = DecoderBlock(c3, c2, c2, deploy=deploy)
        
        # Intermediate dec1_half: Upsamples from 64x128 to 128x256 and fuses with skip_stem (128x256)
        self.dec1_half = DecoderBlock(c2, c1, c1, deploy=deploy)
        
        # Final dec1: Upsamples from 128x256 to 256x512 and fuses with skip_spatial (256x512)
        self.dec1 = DecoderBlock(c1, c1, c1, deploy=deploy)
        
        # Final prediction layer: 1x1 Standard Conv
        self.out_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        # 1. Direct Spatial Detail Injection skip connection (at 256x512)
        skip_spatial = self.spatial_projector(x)
        
        # 2. Stride-4 Early Downsampling
        p1_half = self.stem1(x)  # shape: (B, c1, 128, 256) -> serves as skip_stem
        p1 = self.stem2(p1_half) # shape: (B, c1, 64, 128)
        
        # 3. Encoding Path
        skip2, p2 = self.enc2(p1)  # skip2: (B, c2, 64, 128), p2: (B, c2, 32, 64)
        skip3, p3 = self.enc3(p2)  # skip3: (B, c3, 32, 64), p3: (B, c3, 16, 32)
        skip4, p4 = self.enc4(p3)  # skip4: (B, c4, 16, 32), p4: (B, c4, 8, 16)
        
        # 4. Bottleneck
        b = self.bottleneck(p4)     # b: (B, c5, 8, 16)
        
        # 5. Decoding Path
        d4 = self.dec4(b, skip4)    # d4: (B, c4, 16, 32)
        d3 = self.dec3(d4, skip3)    # d3: (B, c3, 32, 64)
        d2 = self.dec2(d3, skip2)    # d2: (B, c2, 64, 128)
        
        # Intermediate upsampling & fusion at 128x256
        d1_half = self.dec1_half(d2, p1_half) # d1_half: (B, c1, 128, 256)
        
        # Final upsampling & fusion at 256x512
        d1 = self.dec1(d1_half, skip_spatial) # d1: (B, c1, 256, 512)
        
        out = self.out_conv(d1)
        return out

    def switch_to_deploy(self):
        """
        Recursively converts all Reparameterizable convolutions (RepDSConv)
        in the model into deploy mode (fuses parallel branches into single convs).
        """
        if self.deploy:
            return
            
        print("Fusing model branches (switch_to_deploy)...")
        for m in self.modules():
            if m is not self and hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
                
        self.deploy = True
        print("Model branches successfully fused into a clean single-path architecture!")

if __name__ == "__main__":
    # Test model shape and print parameters
    model = DSUnet(in_channels=3, num_classes=4, width_multiplier=0.5, deploy=False)
    x = torch.randn(1, 3, 256, 512)
    model.eval()
    
    # Forward in training mode
    y_train = model(x)
    print(f"Training mode output shape: {y_train.shape}")
    
    # Switch to deploy mode and verify forward
    model.switch_to_deploy()
    y_deploy = model(x)
    print(f"Deploy mode output shape: {y_deploy.shape}")
    
    # Verify outputs match
    diff = torch.max(torch.abs(y_train - y_deploy)).item()
    print(f"Difference between training and deploy outputs: {diff:.8f}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")
