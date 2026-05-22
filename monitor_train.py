import subprocess
import time
import sys
import os
import argparse

def run_training(config_path="configs/default.yaml", resume=False):
    print("=" * 60)
    print("MỞ ĐẦU TIẾN TRÌNH GIÁM SÁT HUẤN LUYỆN (FAULT-TOLERANT MONITOR)")
    print("=" * 60)
    
    cmd = [sys.executable, "train.py", "--config", config_path]
    if resume:
        cmd.append("--resume")
        
    crash_count = 0
    
    while True:
        print(f"\n[Monitor] Đang khởi chạy tiến trình huấn luyện chính: {' '.join(cmd)}")
        print(f"[Monitor] Thời gian bắt đầu: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Khởi chạy train.py và đợi tiến trình kết thúc
            process = subprocess.Popen(cmd)
            process.wait()
            
            return_code = process.returncode
            
            if return_code == 0:
                print("\n" + "=" * 60)
                print("[Monitor] TIẾN TRÌNH HUẤN LUYỆN ĐÃ HOÀN THÀNH THÀNH CÔNG (Exit Code 0).")
                print("=" * 60)
                break
            else:
                crash_count += 1
                print("\n" + "!" * 60)
                print(f"[Monitor] CẢNH BÁO: Tiến trình huấn luyện bị crash (Exit Code: {return_code}).")
                print(f"[Monitor] Số lần crash ghi nhận: {crash_count}")
                print("[Monitor] Tự động khởi động lại sau 5 giây để tiếp tục từ checkpoint...")
                print("!" * 60 + "\n")
                
                # Tự động thêm cờ --resume cho các lần khởi chạy lại sau khi crash
                if "--resume" not in cmd:
                    cmd.append("--resume")
                    
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("[Monitor] Người dùng đã dừng chương trình bằng tổ hợp phím Ctrl+C.")
            print("[Monitor] Tiến trình giám sát kết thúc.")
            print("=" * 60)
            break
        except Exception as e:
            crash_count += 1
            print(f"[Monitor] Lỗi không xác định khi giám sát: {e}")
            print("[Monitor] Thử lại sau 5 giây...")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor DSUnet training and recover from crashes")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
    args = parser.parse_args()
    
    run_training(config_path=args.config, resume=args.resume)
