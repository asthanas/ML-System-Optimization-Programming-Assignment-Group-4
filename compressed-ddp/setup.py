#!/usr/bin/env python3
"""
Platform-agnostic setup script for Compressed-DDP
Works on Linux, macOS, and Windows
"""
import sys
import os
import subprocess
import platform
from pathlib import Path

def run_command(cmd, check=True, quiet=False):
    """Run a command and return success status."""
    try:
        if quiet:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=check,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            result = subprocess.run(cmd, shell=True, check=check)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False

def check_gpu():
    """Check for GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "GPU: Apple Metal (MPS)"
        else:
            return "No GPU detected - CPU mode"
    except ImportError:
        return "PyTorch not yet installed"

def main():
    print("=" * 70)
    print("Compressed-DDP Setup")
    print("=" * 70)
    print(f"\nPlatform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")

    # Check Python version
    if sys.version_info < (3, 9):
        print("\n❌ Error: Python 3.9+ required")
        print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)
    print("✅ Python version OK")

    # Create virtual environment
    venv_dir = Path("venv")
    if not venv_dir.exists():
        print("\n[1/5] Creating virtual environment...")
        if not run_command(f'"{sys.executable}" -m venv venv'):
            print("❌ Failed to create virtual environment")
            sys.exit(1)
        print("✅ Virtual environment created")
    else:
        print("\n[1/5] Virtual environment exists")

    # Determine pip and python paths
    if platform.system() == "Windows":
        pip_cmd = os.path.join("venv", "Scripts", "pip")
        python_cmd = os.path.join("venv", "Scripts", "python")
        activate_msg = "venv\\Scripts\\activate"
    else:
        pip_cmd = os.path.join("venv", "bin", "pip")
        python_cmd = os.path.join("venv", "bin", "python")
        activate_msg = "source venv/bin/activate"

    # Upgrade pip
    print("\n[2/5] Upgrading pip...")
    if not run_command(f'"{pip_cmd}" install --upgrade pip', quiet=True):
        print("⚠️  Warning: Could not upgrade pip (continuing anyway)")
    else:
        print("✅ Pip upgraded")

    # Install requirements
    print("\n[3/5] Installing requirements...")
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        print("   Make sure you're in the compressed-ddp directory")
        sys.exit(1)

    print("   Installing PyTorch and dependencies (this may take a minute)...")
    if not run_command(f'"{pip_cmd}" install -r requirements.txt', quiet=True):
        print("❌ Failed to install requirements")
        print("   Try running manually: pip install -r requirements.txt")
        sys.exit(1)
    print("✅ Requirements installed")

    # Install package in editable mode
    print("\n[4/5] Installing package in editable mode...")
    if not run_command(f'"{pip_cmd}" install -e .', quiet=True):
        print("❌ Failed to install package")
        print("   Try running manually: pip install -e .")
        sys.exit(1)
    print("✅ Package installed")

    # Check GPU availability
    print("\n[5/5] Checking GPU availability...")
    # Try to check GPU using the newly installed torch
    gpu_status = check_gpu()
    if "not yet installed" in gpu_status:
        # Try importing after installation
        sys.path.insert(0, os.path.abspath("venv/lib/python{}.{}/site-packages".format(
            sys.version_info.major, sys.version_info.minor)))
        gpu_status = check_gpu()
    print(f"   {gpu_status}")

    # Success message
    print("\n" + "=" * 70)
    print("✅ Setup complete!")
    print("=" * 70)
    print("\nNext steps:")
    print(f"  1. Activate environment: {activate_msg}")
    print("  2. Run validation: python experiments/quick_validation.py")
    print("  3. Run tests: pytest tests/")
    print("  4. Train model: python train.py --help")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        sys.exit(1)
