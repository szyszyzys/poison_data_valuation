import os
import subprocess
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

def run_cmd(command):
    """Runs a shell command and returns the output."""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "Command failed"

def print_header(title):
    print(f"\n{'='*10} {title} {'='*10}")

def check_cpu():
    print_header("1. CPU CHECK")
    # Get Model Name
    model = run_cmd("cat /proc/cpuinfo | grep 'model name' | uniq | cut -d: -f2")
    # Get Core/Thread Count
    cores = run_cmd("nproc")
    # Get Socket Count (to verify Dual CPU)
    sockets = run_cmd("lscpu | grep 'Socket(s):' | awk '{print $2}'")

    print(f"Paper Claim: Dual Intel(R) Xeon(R) Gold 5317 (24 cores/48 threads total)")
    print(f"Detected:    {sockets} x {model.strip()}")
    print(f"Total Threads: {cores}")

def check_ram():
    print_header("2. RAM CHECK")
    # Get total memory in GiB
    mem_gb = run_cmd("free -g | grep Mem | awk '{print $2}'")

    print(f"Paper Claim: 503 GiB System Memory")
    print(f"Detected:    {mem_gb} GiB (Note: 'free' often shows slightly less than physical due to kernel reservation)")

def check_gpu():
    print_header("3. GPU CHECK")
    print(f"Paper Claim: 8x NVIDIA RTX A6000 (48 GB VRAM)")

    if not torch.cuda.is_available():
        print("❌ CUDA not available! Cannot check GPUs.")
        return

    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    # Get total memory of device 0 in GB
    mem_bytes = torch.cuda.get_device_properties(0).total_memory
    mem_gb = mem_bytes / (1024**3)

    print(f"Detected Count: {count}")
    print(f"Detected Model: {name}")
    print(f"Detected VRAM:  {mem_gb:.2f} GB")

def check_software():
    print_header("4. SOFTWARE CHECK")
    os_ver = run_cmd("lsb_release -d | cut -f2")

    print(f"Paper Claim: CUDA 12.4, PyTorch 2.x, Ubuntu 22.04")
    print(f"OS Detected:      {os_ver}")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"CUDA Version:     {torch.version.cuda}")

def benchmark_runtime():
    print_header("5. RUNTIME BASELINE CHECK")
    print("Simulating training load to verify '4.5 GPU-hours' claim...")

    # Setup Dummy Data (CIFAR-100 size: Batch 64, 3 channels, 32x32 image)
    batch_size = 64
    input_tensor = torch.randn(batch_size, 3, 32, 32).cuda()
    target = torch.randint(0, 100, (batch_size,)).cuda()

    # Setup Model (ResNet18)
    model = resnet18(num_classes=100).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Measure 100 steps
    steps = 100
    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end = time.time()

    avg_step_time = (end - start) / steps

    # Math for Paper Claim
    # Paper: 30 sellers, 100 global rounds.
    # Assumption: Each seller trains for E local epochs.
    # CIFAR-100 has 50k images. 30 sellers = ~1666 images per seller.
    # Batch 64 ~= 26 steps per epoch per seller.

    images_per_seller = 50000 / 30
    steps_per_epoch = int(images_per_seller / batch_size)
    local_epochs = 2 # Assuming standard setting

    total_steps_per_round = steps_per_epoch * local_epochs * 30
    total_experiment_steps = total_steps_per_round * 100 # 100 global rounds

    estimated_seconds = total_experiment_steps * avg_step_time
    estimated_hours = estimated_seconds / 3600

    print(f"Avg time per batch (ResNet18/CIFAR100): {avg_step_time*1000:.2f} ms")
    print(f"Estimated Total Time (30 sellers, 2 local epochs, 100 rounds): {estimated_hours:.2f} Hours")
    print(f"Paper Claim: ~4.5 Hours")

    if 3.5 < estimated_hours < 5.5:
        print("✅ Runtime claim looks plausible.")
    else:
        print("⚠️  Runtime deviation. Check your 'Local Epochs' or 'Batch Size' assumptions.")

if __name__ == "__main__":
    check_cpu()
    check_ram()
    check_gpu()
    check_software()
    benchmark_runtime()