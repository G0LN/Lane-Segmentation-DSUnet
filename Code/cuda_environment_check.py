import torch
import cv2
import sklearn

print(f"PyTorch Version: {torch.__version__}")
print(f"OpenCV Version: {cv2.__version__}")
print(f"Scikit-learn Version: {sklearn.__version__}")

if torch.cuda.is_available():
    print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
    print("Environment is ready for training!")
else:
    print("❌ WARN : PyTorch can not find out GPU!")