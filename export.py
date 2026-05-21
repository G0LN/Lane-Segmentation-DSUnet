import argparse
import yaml
import torch
from thop import profile
from thop import clever_format
from models import DSUnet

def export_and_profile(config_path, output_onnx="dsunet.onnx"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cpu') # Export usually done on CPU
    num_classes = config['model']['num_classes']
    in_channels = config['model']['in_channels']
    img_h = config['dataset']['image_height']
    img_w = config['dataset']['image_width']
    
    # Init model
    model = DSUnet(
        in_channels=in_channels, 
        num_classes=num_classes,
        dropout=config['model'].get('dropout', 0.5)
    ).to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, in_channels, img_h, img_w).to(device)
    
    print("="*50)
    print(f"Model: {config['model']['name']}")
    print(f"Input Shape: (1, {in_channels}, {img_h}, {img_w})")
    
    # Calculate FLOPs and Params using THOP
    print("\nCalculating FLOPs and Parameters...")
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    
    # THOP returns MACs (Multiply-Accumulate operations), FLOPs is typically 2 * MACs
    flops = 2 * macs
    
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
    flops_formatted, _ = clever_format([flops, params], "%.3f")
    
    print("-" * 50)
    print(f"Total Parameters: {params_formatted} ({params} params)")
    print(f"MACs: {macs_formatted}")
    print(f"FLOPs: {flops_formatted}")
    print("=" * 50)
    
    # Export to ONNX
    print(f"\nExporting model to ONNX format: {output_onnx}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_onnx, 
        export_params=True, 
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Export successful! File saved at: {output_onnx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile and Export DSUnet")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='dsunet.onnx', help='Path to output ONNX file')
    args = parser.parse_args()
    
    export_and_profile(args.config, args.output)
