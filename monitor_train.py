import subprocess
import time
import sys
import os

def run_training():
    print("=" * 60)
    print("MỞ ĐẦU TIẾN TRÌNH GIÁM SÁT HUẤN LUYỆN (FAULT-TOLERANT MONITOR)")
    print("=" * 60)
    
    config_path = "configs/default.yaml"
    cmd = [sys.executable, "train.py"]
    
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
    run_training()
