import subprocess
import time
import sys
import os
import argparse

def run_training(config_path="configs/default.yaml", resume=False):
    print("=" * 60)
    print("      FAULT-TOLERANT TRAINING MONITOR PROCESS      ")
    print("=" * 60)
    
    cmd = [sys.executable, "train.py", "--config", config_path]
    if resume:
        cmd.append("--resume")
        
    crash_count = 0
    
    while True:
        print(f"\n[Monitor] Launching main training process: {' '.join(cmd)}")
        print(f"[Monitor] Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Start train.py and wait for it to finish
            process = subprocess.Popen(cmd)
            process.wait()
            
            return_code = process.returncode
            
            if return_code == 0:
                print("\n" + "=" * 60)
                print("[Monitor] TRAINING COMPLETED SUCCESSFULLY (Exit Code 0).")
                print("=" * 60)
                break
            else:
                crash_count += 1
                print("\n" + "!" * 60)
                print(f"[Monitor] WARNING: Training process crashed (Exit Code: {return_code}).")
                print(f"[Monitor] Total recorded crashes: {crash_count}")
                print("[Monitor] Automatically restarting in 5 seconds to resume from last checkpoint...")
                print("!" * 60 + "\n")
                
                # Auto-append --resume flag for restart runs after crash
                if "--resume" not in cmd:
                    cmd.append("--resume")
                    
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("[Monitor] Training stopped by user (Ctrl+C).")
            print("[Monitor] Monitoring process terminated.")
            print("=" * 60)
            break
        except Exception as e:
            crash_count += 1
            print(f"[Monitor] Unknown error during monitoring: {e}")
            print("[Monitor] Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor B-DSUnet training and recover from crashes")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
    args = parser.parse_args()
    
    run_training(config_path=args.config, resume=args.resume)
