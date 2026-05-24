import torch
import torch.nn as nn
from .components.encoder import EncoderBlock
from .components.decoder import DecoderBlock

class DSUnet(nn.Module):
    """
    Bilateral DSUnet (B-DSUnet) Architecture for Lane Detection / Segmentation.
    Optimized with a Stem Block for early downsampling (saving massive FLOPs),
    a Direct Spatial Detail Injection Branch to preserve high-resolution boundary details,
    an Asymmetric Decoder, and Skip Connection Compression.
    """
    def __init__(self, in_channels=3, num_classes=4, dropout=0.5, width_multiplier=1.0, deploy=False):
        super(DSUnet, self).__init__()
        self.deploy = deploy
        
        # Scale number of channels dynamically based on width_multiplier (alpha)
        c1 = int(64 * width_multiplier)
        c2 = int(128 * width_multiplier)
        c3 = int(256 * width_multiplier)
        c4 = int(512 * width_multiplier)
        c5 = int(1024 * width_multiplier) # Bottleneck
        
        # 1. Stem Block: Immediately downsamples the input image by 2x (from 256x512 to 128x256)
        # using a fast Conv 3x3 with stride=2. This saves massive FLOPs for the entire encoder!
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Direct Spatial Detail Injection Branch:
        # Projects raw high-res image (256x512) directly to c1 channels to serve as
        # skip connection detail for the final decoder block dec1, bypassing the entire Encoder.
        # This preserves crisp boundaries of thin objects (like lanes) with negligible FLOPs.
        self.spatial_projector = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # 3. Encoder Path (Starts directly at 128x256 resolution)
        self.enc2 = EncoderBlock(c1, c2, use_pool=True, deploy=deploy)
        self.enc3 = EncoderBlock(c2, c3, use_pool=True, deploy=deploy)
        self.enc4 = EncoderBlock(c3, c4, use_pool=True, dropout_prob=dropout, deploy=deploy)
        
        # Bottleneck (No pooling, 16x32 resolution)
        self.bottleneck = EncoderBlock(c4, c5, use_pool=False, dropout_prob=dropout, deploy=deploy)
        
        # 4. Decoder Path (Upsampling and fusing features)
        self.dec4 = DecoderBlock(c5, c4, c4, dropout_prob=dropout, deploy=deploy)
        self.dec3 = DecoderBlock(c4, c3, c3, deploy=deploy)
        self.dec2 = DecoderBlock(c3, c2, c2, deploy=deploy)
        self.dec1 = DecoderBlock(c2, c1, c1, deploy=deploy)
        
        # Final prediction layer: 1x1 Standard Conv
        self.out_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        # 1. Direct Spatial Detail Injection skip connection (at 256x512)
        skip_spatial = self.spatial_projector(x)
        
        # 2. Early Downsampling via Stem (to 128x256)
        p1 = self.stem(x)
        
        # 3. Encoding Path
        skip2, p2 = self.enc2(p1)
        skip3, p3 = self.enc3(p2)
        skip4, p4 = self.enc4(p3)
        
        # 4. Bottleneck
        b = self.bottleneck(p4)
        
        # 5. Decoding Path
        d4 = self.dec4(b, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip_spatial) # Fuse with high-res injected spatial features!
        
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
    model = DSUnet(in_channels=3, num_classes=4, deploy=False)
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
