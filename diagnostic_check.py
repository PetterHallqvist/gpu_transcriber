#!/usr/bin/env python3
"""
Transcription Environment Diagnostic Tool
=========================================
Checks GPU status, environment setup, and identifies issues
that might cause slow CPU-mode transcription.
"""

import sys
import os
import time
import subprocess
import json
from datetime import datetime

def log(message, level="INFO"):
    """Simple logging with timestamps"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "DEBUG": "🔍"
    }.get(level, "  ")
    print(f"[{timestamp}] {prefix} {message}")

def check_python_environment():
    """Check Python and PyTorch setup"""
    log("Checking Python Environment", "DEBUG")
    log(f"Python version: {sys.version}")
    log(f"Python executable: {sys.executable}")
    
    try:
        import torch
        log(f"PyTorch version: {torch.__version__}", "SUCCESS")
        log(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            log(f"CUDA version: {torch.version.cuda}")
            log(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        else:
            log("CUDA not available in PyTorch", "WARNING")
        
        return torch.cuda.is_available()
    except ImportError as e:
        log(f"PyTorch not available: {e}", "ERROR")
        return False

def check_nvidia_drivers():
    """Check NVIDIA driver status"""
    log("Checking NVIDIA Drivers", "DEBUG")
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            log("nvidia-smi working", "SUCCESS")
            
            # Get GPU info
            gpu_result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.total,driver_version,temperature.gpu', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if gpu_result.returncode == 0:
                for line in gpu_result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            name, memory, driver, temp = parts[:4]
                            log(f"GPU: {name}")
                            log(f"  Memory: {memory} MB")
                            log(f"  Driver: {driver}")
                            log(f"  Temperature: {temp}°C")
            
            return True
        else:
            log(f"nvidia-smi failed: {result.stderr}", "ERROR")
            return False
            
    except FileNotFoundError:
        log("nvidia-smi not found", "ERROR")
        return False
    except subprocess.TimeoutExpired:
        log("nvidia-smi timeout", "ERROR")
        return False
    except Exception as e:
        log(f"nvidia-smi error: {e}", "ERROR")
        return False

def check_instance_type():
    """Check if we're on the right instance type"""
    log("Checking Instance Type", "DEBUG")
    
    try:
        # Check instance metadata
        result = subprocess.run([
            'curl', '-s', '--connect-timeout', '3',
            'http://169.254.169.254/latest/meta-data/instance-type'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            instance_type = result.stdout.strip()
            log(f"Instance type: {instance_type}")
            
            if instance_type.startswith('g4dn'):
                log("GPU instance detected", "SUCCESS")
                return True
            else:
                log(f"Non-GPU instance: {instance_type}", "WARNING")
                log("GPU transcription requires g4dn.xlarge or similar", "WARNING")
                return False
        else:
            log("Could not determine instance type", "WARNING")
            return None
            
    except Exception as e:
        log(f"Instance type check failed: {e}", "WARNING")
        return None

def check_transcription_environment():
    """Check transcription directory structure"""
    log("Checking Transcription Environment", "DEBUG")
    
    paths = {
        "/opt/transcribe": "Base directory",
        "/opt/transcribe/venv": "Virtual environment", 
        "/opt/transcribe/models": "Model cache directory",
        "/opt/transcribe/cache": "Cache directory",
        "/opt/transcribe/scripts": "Scripts directory",
        "/opt/transcribe/cache/cache_info.json": "Cache metadata",
        "/opt/transcribe/scripts/transcribe_optimized.py": "Main script",
        "/opt/transcribe/scripts/api_server.py": "API server"
    }
    
    all_good = True
    
    for path, description in paths.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                count = len(os.listdir(path))
                log(f"{description}: ✓ ({count} items)")
            else:
                size = os.path.getsize(path)
                log(f"{description}: ✓ ({size} bytes)")
        else:
            log(f"{description}: MISSING", "ERROR")
            all_good = False
    
    return all_good

def check_cache_status():
    """Check model cache status"""
    log("Checking Model Cache", "DEBUG")
    
    cache_file = "/opt/transcribe/cache/cache_info.json"
    
    if not os.path.exists(cache_file):
        log("Cache info file missing", "WARNING")
        return False
    
    try:
        with open(cache_file, 'r') as f:
            cache_info = json.load(f)
        
        log("Cache configuration:", "SUCCESS")
        for key, value in cache_info.items():
            log(f"  {key}: {value}")
        
        # Check if models exist
        cache_dir = cache_info.get("cache_dir", "/opt/transcribe/models")
        if os.path.exists(cache_dir):
            model_files = os.listdir(cache_dir)
            log(f"Model files: {len(model_files)} files in cache")
            
            # Check for specific model files
            expected_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            found_files = [f for f in expected_files if any(f in mf for mf in model_files)]
            log(f"Key model files found: {found_files}")
            
            return len(found_files) > 0
        else:
            log(f"Model cache directory missing: {cache_dir}", "ERROR")
            return False
            
    except Exception as e:
        log(f"Error reading cache file: {e}", "ERROR")
        return False

def check_services():
    """Check systemd services"""
    log("Checking System Services", "DEBUG")
    
    try:
        # Check transcribe-api service
        result = subprocess.run([
            'systemctl', 'is-enabled', 'transcribe-api'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            log("transcribe-api service: enabled", "SUCCESS")
        else:
            log("transcribe-api service: not enabled", "WARNING")
        
        # Check service status
        status_result = subprocess.run([
            'systemctl', 'is-active', 'transcribe-api'
        ], capture_output=True, text=True)
        
        status = status_result.stdout.strip()
        if status == "active":
            log("transcribe-api service: running", "SUCCESS")
        else:
            log(f"transcribe-api service: {status}", "WARNING")
    
    except Exception as e:
        log(f"Service check failed: {e}", "WARNING")

def run_quick_gpu_test():
    """Run a quick GPU test"""
    log("Running Quick GPU Test", "DEBUG")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            log("CUDA not available - skipping GPU test", "WARNING")
            return False
        
        log("Testing GPU memory allocation...")
        device = torch.device('cuda')
        
        # Test basic GPU operations
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        
        log("GPU memory allocation: SUCCESS", "SUCCESS")
        
        # Check memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_cached = torch.cuda.memory_reserved() / 1024**2
        log(f"GPU memory - Allocated: {memory_allocated:.1f} MB, Cached: {memory_cached:.1f} MB")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        log(f"GPU test failed: {e}", "ERROR")
        return False

def suggest_fixes(results):
    """Suggest fixes based on diagnostic results"""
    log("Diagnostic Summary & Recommendations", "DEBUG")
    
    if not results['nvidia_drivers']:
        log("🔧 Fix 1: Install NVIDIA drivers", "WARNING")
        log("   sudo apt update && sudo apt install nvidia-driver-535-server")
        log("   sudo reboot")
    
    if not results['pytorch_cuda']:
        log("🔧 Fix 2: Install CUDA-enabled PyTorch", "WARNING") 
        log("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    if not results['environment']:
        log("🔧 Fix 3: Setup transcription environment", "WARNING")
        log("   Run: ./build_ami.sh to create proper AMI")
    
    if not results['cache']:
        log("🔧 Fix 4: Build model cache", "WARNING")
        log("   cd /opt/transcribe && source venv/bin/activate && python advanced_cache.py")
    
    if results['instance_type'] == False:
        log("🔧 Fix 5: Use GPU instance", "WARNING")
        log("   Launch g4dn.xlarge instance instead of current type")
    
    # Overall recommendation
    if all([results['nvidia_drivers'], results['pytorch_cuda'], results['environment'], results['cache']]):
        log("✅ Environment looks good! GPU transcription should work.", "SUCCESS")
    else:
        log("❌ Issues found - transcription will be slow on CPU", "ERROR")
        log("💡 Consider rebuilding AMI: ./build_ami.sh", "WARNING")

def main():
    log("🔍 Transcription Environment Diagnostics")
    log("=" * 50)
    
    results = {}
    
    # Run all checks
    results['instance_type'] = check_instance_type()
    results['nvidia_drivers'] = check_nvidia_drivers()
    results['pytorch_cuda'] = check_python_environment()
    results['environment'] = check_transcription_environment()
    results['cache'] = check_cache_status()
    check_services()
    results['gpu_test'] = run_quick_gpu_test()
    
    log("=" * 50)
    suggest_fixes(results)
    
    # Exit code for scripting
    if all([results['nvidia_drivers'], results['pytorch_cuda'], results['environment']]):
        log("🎯 Ready for fast GPU transcription!", "SUCCESS")
        sys.exit(0)
    else:
        log("⚠️  Will fall back to slow CPU transcription", "WARNING")
        sys.exit(1)

if __name__ == "__main__":
    main() 