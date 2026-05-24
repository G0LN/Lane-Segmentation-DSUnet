import torch
import sys
import os

# Add workspace to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.losses import get_criterion

def test_joint_loss():
    print("Running JointCEDiceLoss Unit Test...")
    torch.manual_seed(42)
    
    num_classes = 9
    batch_size = 2
    height, width = 64, 128
    
    # Initialize joint loss using our utility
    device = torch.device("cpu")
    criterion = get_criterion("JointCEDiceLoss", num_classes=num_classes, device=device)
    
    # Generate mock predictions (logits: raw unnormalized scores) and ground truth masks
    logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size, height, width), dtype=torch.long)
    
    # Calculate loss
    loss = criterion(logits, targets)
    print(f"-> Calculated Loss: {loss.item():.4f}")
    
    # Verify it is a valid positive scalar
    assert loss.item() > 0, "Loss value must be positive!"
    
    # Verify backpropagation works (requires_grad is preserved through the loss)
    loss.backward()
    grad = logits.grad
    assert grad is not None, "Gradients must not be None after backpropagation!"
    print(f"-> Gradient max value: {grad.max().item():.6f}")
    print("[PASS] JointCEDiceLoss Unit Test PASSED successfully!\n")

if __name__ == "__main__":
    test_joint_loss()
