import subprocess
import sys
import os
import platform

# 1. Get Python version
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# 2. Get installed pip packages
installed = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
torch_related = [line for line in installed.splitlines() if line.lower().startswith("torch") or "audio" in line.lower() or "vision" in line.lower()]

# 3. Create environment file
env_lines = [
    "name: torch-env-export",
    "channels:",
    "  - pytorch",
    "  - nvidia",
    "  - conda-forge",
    "  - defaults",
    "dependencies:",
    f"  - python={python_version}",
    "  - pip",
    "  - pip:",
]

for line in torch_related:
    env_lines.append(f"      - {line}")

output_file = "environment-torch.yml"
with open(output_file, "w") as f:
    f.write("\n".join(env_lines))

print(f"✅ Torch environment written to {output_file}")
print("\n".join(env_lines))
