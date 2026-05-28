import os
import sys
import time
import yaml
import torch
import numpy as np
import onnxruntime as ort

# Add the root directory to path to ensure all modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import DSUnet

def get_file_size_mb(path):
    if not os.path.exists(path):
        return 0.0
    size = os.path.getsize(path)
    # Check if there is an external data file (common in newer PyTorch ONNX exports)
    if os.path.exists(path + ".data"):
        size += os.path.getsize(path + ".data")
    return size / (1024 * 1024)

def run_pytorch_benchmark(config, device_name, num_runs, warmup_runs):
    device = torch.device(device_name)
    img_h = config['dataset']['image_height']
    img_w = config['dataset']['image_width']
    in_channels = config['model']['in_channels']
    num_classes = config['model']['num_classes']
    width_multiplier = config['model'].get('width_multiplier', 0.5)
    dropout = config['model'].get('dropout', 0.2)
    
    # Init model
    model = DSUnet(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
        width_multiplier=width_multiplier,
        deploy=False
    ).to(device)
    
    model.eval()
    model.switch_to_deploy() # Benchmark actual deploy/fused state
    
    dummy_input = torch.randn(1, in_channels, img_h, img_w).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            if device_name == "cuda":
                torch.cuda.synchronize()
                
    # Memory tracking setup
    if device_name == "cuda":
        torch.cuda.reset_peak_memory_stats()
        
    # Timing
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            if device_name == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0) # ms
            
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency
    
    memory_str = "N/A"
    if device_name == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        memory_str = f"{peak_mem:.2f} MB"
        
    return {
        "precision": "FP32",
        "size": "N/A (PyTorch)",
        "latency": avg_latency,
        "std": std_latency,
        "fps": fps,
        "memory": memory_str
    }

def run_onnx_benchmark(model_path, num_runs, warmup_runs):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # Optimized thread pool representing Raspberry Pi 5 CPU
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    
    # Batch dimension fix
    if isinstance(input_shape[0], str) or input_shape[0] < 0:
        input_shape[0] = 1
        
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup_runs):
        _ = session.run([output_name], {input_name: dummy_input})
        
    # Timing
    latencies = []
    for _ in range(num_runs):
        t_start = time.perf_counter()
        _ = session.run([output_name], {input_name: dummy_input})
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000.0) # ms
        
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency
    size_mb = get_file_size_mb(model_path)
    
    precision = "INT8" if "int8" in model_path.lower() else "FP32"
    
    return {
        "precision": precision,
        "size": f"{size_mb:.2f} MB",
        "latency": avg_latency,
        "std": std_latency,
        "fps": fps,
        "memory": "N/A (CPU)"
    }

def main():
    config_path = "configs/default.yaml"
    num_runs = 200
    warmup_runs = 20
    
    print("=" * 80)
    print("              UNIFIED B-DSUNET MULTI-ENGINE BENCHMARK TOOL              ")
    print("=" * 80)
    print(f"[Info] Running benchmarks: warmup = {warmup_runs} runs | profiling = {num_runs} runs")
    
    if not os.path.exists(config_path):
        print(f"[Error] Config file '{config_path}' not found.")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    results = {}
    
    # 1. Run PyTorch CPU Benchmark
    print("\n[Profiling] Running PyTorch CPU Benchmark...")
    try:
        results["PyTorch CPU"] = run_pytorch_benchmark(config, "cpu", num_runs, warmup_runs)
        print(f"  -> Speed: {results['PyTorch CPU']['fps']:.2f} FPS")
    except Exception as e:
        print(f"  -> [Failed]: {e}")
        
    # 2. Run PyTorch CUDA Benchmark (if GPU available)
    if torch.cuda.is_available():
        print("\n[Profiling] Running PyTorch CUDA GPU Benchmark...")
        try:
            results["PyTorch CUDA"] = run_pytorch_benchmark(config, "cuda", num_runs, warmup_runs)
            print(f"  -> Speed: {results['PyTorch CUDA']['fps']:.2f} FPS | Peak VRAM: {results['PyTorch CUDA']['memory']}")
        except Exception as e:
            print(f"  -> [Failed]: {e}")
            
    # 3. Run ONNX FP32 CPU Benchmark
    onnx_fp32_path = "dsunet.onnx"
    if os.path.exists(onnx_fp32_path):
        print(f"\n[Profiling] Running ONNX Runtime FP32 CPU Benchmark ({onnx_fp32_path})...")
        try:
            results["ONNX FP32 CPU"] = run_onnx_benchmark(onnx_fp32_path, num_runs, warmup_runs)
            print(f"  -> Speed: {results['ONNX FP32 CPU']['fps']:.2f} FPS")
        except Exception as e:
            print(f"  -> [Failed]: {e}")
    else:
        print(f"\n[Info] Skipping ONNX FP32 CPU: file '{onnx_fp32_path}' does not exist.")
        
    # 4. Run ONNX INT8 CPU Benchmark
    onnx_int8_path = "dsunet_int8.onnx"
    if os.path.exists(onnx_int8_path):
        print(f"\n[Profiling] Running ONNX Runtime INT8 Quantized CPU Benchmark ({onnx_int8_path})...")
        try:
            results["ONNX INT8 CPU"] = run_onnx_benchmark(onnx_int8_path, num_runs, warmup_runs)
            print(f"  -> Speed: {results['ONNX INT8 CPU']['fps']:.2f} FPS")
        except Exception as e:
            print(f"  -> [Failed]: {e}")
    else:
        print(f"\n[Info] Skipping ONNX INT8 CPU: file '{onnx_int8_path}' does not exist.")
        
    # 5. Output Unified Side-by-Side Comparison Report
    print("\n" + "=" * 80)
    print("                     SYSTEM-WIDE PERFORMANCE COMPARISON REPORT                  ")
    print("=" * 80)
    print(f"{'Inference Engine':<18} | {'Precision':<9} | {'File Size':<12} | {'Avg Latency':<12} | {'Speed (FPS)':<12} | {'VRAM/RAM':<10}")
    print("-" * 80)
    
    base_fps = results.get("PyTorch CPU", {}).get("fps", 1.0)
    
    for engine, res in results.items():
        rel_speedup = res['fps'] / base_fps
        speedup_str = f" ({rel_speedup:.2f}x)" if engine != "PyTorch CPU" else " (Base)"
        
        print(f"{engine:<18} | {res['precision']:<9} | {res['size']:<12} | {res['latency']:6.2f} ms   | {res['fps']:6.2f} FPS {speedup_str:<8} | {res['memory']:<10}")
    print("-" * 80)
    print("[Note] ONNX FP32/INT8 runs with 'intra_op_num_threads=4' to simulate Pi 5 CPU.")
    print("[Note] On x86 PC, INT8 may be slower than FP32 unless CPU supports AVX-512 VNNI.")
    print("[Note] On ARM CPU (Raspberry Pi 5), INT8 will be significantly faster than FP32!")
    print("=" * 80)

if __name__ == "__main__":
    main()
