import os
import sys
from onnxruntime.quantization import quantize_dynamic, QuantType

# Add parent directory to sys.path to allow imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def quantize_model(input_model="dsunet.onnx", output_model="dsunet_int8.onnx"):
    print("=" * 60)
    print("           B-DSUnet INT8 QUANTIZATION TOOL            ")
    print("=" * 60)
    
    if not os.path.exists(input_model):
        print(f"[Error] Source model '{input_model}' not found.")
        print("[Info] Please run 'python tools/export.py' first.")
        return
        
    # Calculate initial size
    size_fp32_base = os.path.getsize(input_model)
    size_fp32_data = os.path.getsize(input_model + ".data") if os.path.exists(input_model + ".data") else 0
    total_size_fp32 = size_fp32_base + size_fp32_data
    
    print(f"[Info] Loading FP32 model: {input_model} ({total_size_fp32 / (1024*1024):.2f} MB)")
    print("[Info] Quantizing model weights from FP32 to INT8...")
    
    try:
        quantize_dynamic(
            model_input=input_model,
            model_output=output_model,
            weight_type=QuantType.QUInt8
        )
        print(f"[SUCCESS] Quantized model saved at: {output_model}")
        
        # Calculate compressed size
        size_int8_base = os.path.getsize(output_model)
        size_int8_data = os.path.getsize(output_model + ".data") if os.path.exists(output_model + ".data") else 0
        total_size_int8 = size_int8_base + size_int8_data
        
        print("-" * 60)
        print(f"  - FP32 Model Size: {total_size_fp32 / (1024*1024):.2f} MB")
        print(f"  - INT8 Model Size: {total_size_int8 / (1024*1024):.2f} MB")
        print(f"  - Compression Ratio: {total_size_fp32 / total_size_int8:.2f}x smaller!")
        print("-" * 60)
        print("[Next Step] You can copy 'dsunet_int8.onnx' directly to your Raspberry Pi 5")
        print("and run inference using ONNX Runtime CPU. It will be incredibly fast!")
        
    except Exception as e:
        print(f"[Error] Quantization failed: {e}")
        
    print("=" * 60)

if __name__ == "__main__":
    # If run from project root, target models in project root
    quantize_model()
