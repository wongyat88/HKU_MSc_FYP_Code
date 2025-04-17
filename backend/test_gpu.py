import torch


# Add this at the beginning of your script to diagnose the issue
def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU is available - Found {torch.cuda.device_count()} device(s)")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
    else:
        print("No GPU available - Using CPU")
        print(f"PyTorch version: {torch.__version__}")


check_gpu()
