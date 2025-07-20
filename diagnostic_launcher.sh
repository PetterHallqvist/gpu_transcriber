#!/bin/bash

# Diagnostic Instance Launcher
# Launches a transcription instance and runs comprehensive diagnostics

echo "🔍 Diagnostic Instance Launcher"
echo "================================"
echo "This will launch an instance and run diagnostics"

export AWS_DEFAULT_REGION=eu-north-1

# Cleanup function for automatic instance termination
cleanup_instances() {
    echo ""
    echo "Diagnostic cleanup..."
    
    if [ ! -z "$INSTANCE_ID" ]; then
        echo "Terminating diagnostic instance: $INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null 2>&1
    fi
    
    # Clean up temp files
    rm -f /tmp/diagnostic_instance_id
    
    echo "Diagnostic cleanup completed"
}

# Set up signal traps for automatic cleanup
trap cleanup_instances EXIT INT TERM

# Clean up any previous diagnostic instances
if [ -f /tmp/diagnostic_instance_id ]; then
    OLD_INSTANCE_ID=$(cat /tmp/diagnostic_instance_id 2>/dev/null)
    if [ ! -z "$OLD_INSTANCE_ID" ]; then
        echo "Cleaning up previous diagnostic instance: $OLD_INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$OLD_INSTANCE_ID" >/dev/null 2>&1
    fi
fi

echo ""
echo "Loading AMI..."

# Check for AMI
if [ -f ami_id.txt ]; then
    AMI_ID=$(cat ami_id.txt)
    echo "Using AMI: $AMI_ID"
else
    echo "ERROR: No AMI found!"
    echo "Please run: ./build_ami.sh first"
    exit 1
fi

echo ""
echo "Launching diagnostic instance..."

# Check for existing instances first
echo "Checking for existing instances..."
EXISTING_INSTANCE=$(aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
              "Name=instance-type,Values=g4dn.xlarge" \
              "Name=instance.group-name,Values=transcription-g4dn-sg" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text 2>/dev/null)

if [ "$EXISTING_INSTANCE" != "None" ] && [ ! -z "$EXISTING_INSTANCE" ] && [ "$EXISTING_INSTANCE" != "null" ]; then
    echo "Found existing instance: $EXISTING_INSTANCE"
    echo "Using existing instance for diagnostics..."
    INSTANCE_ID="$EXISTING_INSTANCE"
    
    # Get the public IP of existing instance
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    echo "Instance IP: $PUBLIC_IP"
    echo "Skipping launch - going directly to diagnostics..."
    
    # Store for cleanup
    echo "$INSTANCE_ID" > /tmp/diagnostic_instance_id
    
else
    echo "Launching new diagnostic instance..."

    # Start timer for launch performance
    START_TIME=$(date +%s)

    # Try multiple zones for instance availability
    ZONES=("eu-north-1a" "eu-north-1b" "eu-north-1c")
    INSTANCE_ID=""
    zone_attempt=0

    for zone in "${ZONES[@]}"; do
        zone_attempt=$((zone_attempt + 1))
        echo "Trying launch in zone: $zone (attempt $zone_attempt/3)"
        
        # Minimal delay between attempts
        if [ $zone_attempt -gt 1 ]; then
            echo "Waiting 3 seconds before next zone..."
            sleep 3
        fi
        
        INSTANCE_OUTPUT=$(aws ec2 run-instances \
            --image-id "$AMI_ID" \
            --instance-type "g4dn.xlarge" \
            --key-name "transcription-ec2" \
            --security-groups "transcription-g4dn-sg" \
            --placement "AvailabilityZone=$zone" \
            --block-device-mappings '[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": 30,
                        "VolumeType": "gp3",
                        "Iops": 3000,
                        "Throughput": 125,
                        "DeleteOnTermination": true
                    }
                }
            ]' \
            --count 1 \
            --output text \
            --query 'Instances[0].InstanceId' 2>&1)
        
        INSTANCE_ID=$(echo "$INSTANCE_OUTPUT" | grep -v "^An error occurred" | head -1)
        
        if [ ! -z "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
            echo "SUCCESS: Instance launched in zone: $zone"
            echo "Instance ID: $INSTANCE_ID"
            break
        else
            echo "FAILED: Failed in zone $zone"
            if echo "$INSTANCE_OUTPUT" | grep -q "An error occurred"; then
                echo "Error: $INSTANCE_OUTPUT"
            fi
            echo "Trying next zone..."
        fi
    done

    if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
        echo "ERROR: Failed to create instance in any zone"
        echo "This might be due to:"
        echo "  - G and VT Instance quota exceeded (g4dn.xlarge uses this quota)"
        echo "  - High demand in all zones"
        echo "  - Try again in a few minutes"
        exit 1
    fi

    # Store for cleanup
    echo "$INSTANCE_ID" > /tmp/diagnostic_instance_id

    echo "Instance launched: $INSTANCE_ID"

    # Wait for running state
    echo "Waiting for instance to boot..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID

    # Get IP immediately
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    echo "Public IP: $PUBLIC_IP"

    # SSH wait
    echo "Waiting for SSH (target <10s)..."
    SSH_READY=false
    for i in {1..15}; do
        if ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no -o ConnectTimeout=3 ubuntu@$PUBLIC_IP "echo 'SSH Ready'" >/dev/null 2>&1; then
            SSH_READY=true
            echo "SSH ready!"
            break
        fi
        echo "Attempt $i/15: SSH connecting..."
        sleep 2
    done

    if [ "$SSH_READY" = false ]; then
        echo "ERROR: SSH failed"
        exit 1
    fi

    # Calculate boot time
    BOOT_TIME=$(($(date +%s) - START_TIME))
    echo "Boot completed in ${BOOT_TIME}s"

fi  # End of new instance launch

# Verify SSH connection
echo ""
echo "Verifying SSH connection..."
SSH_READY=false
for i in {1..5}; do
    if ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no -o ConnectTimeout=3 ubuntu@$PUBLIC_IP "echo 'SSH Verified'" >/dev/null 2>&1; then
        SSH_READY=true
        echo "SSH connection verified!"
        break
    fi
    echo "Attempt $i/5: Verifying SSH..."
    sleep 1
done

if [ "$SSH_READY" = false ]; then
    echo "ERROR: SSH connection failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "🔧 INSTALLING CRITICAL FIXES FIRST"
echo "=================================================="

# Upload diagnostic script
echo "Uploading diagnostic tools..."
scp -i transcription-ec2.pem -o StrictHostKeyChecking=no diagnostic_check.py ubuntu@$PUBLIC_IP:/tmp/

echo "Installing NVIDIA drivers, CUDA PyTorch, and building model cache..."
echo "This may take 5-10 minutes..."

# Install all the critical fixes first
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'INSTALL_FIXES_EOF'
set -e

echo "🔧 INSTALLING CRITICAL FIXES"
echo "============================="

# Check if we need to install drivers
if nvidia-smi >/dev/null 2>&1; then
    echo "✅ NVIDIA drivers already working"
else
    echo "🔧 Installing NVIDIA drivers..."
    
    # Update package list
    sudo apt update -y
    
    # Install NVIDIA drivers with better approach
    echo "Installing ubuntu-drivers-common..."
    sudo apt install -y ubuntu-drivers-common
    
    echo "Auto-detecting and installing NVIDIA drivers..."
    sudo ubuntu-drivers autoinstall
    
    # Also try specific driver as fallback
    echo "Installing specific driver as backup..."
    sudo apt install -y nvidia-driver-535-server nvidia-utils-535-server
    
    echo "⚠️  Driver installation complete - reboot may be required"
    echo "Testing nvidia-smi..."
    
    # Test nvidia-smi after driver installation
    if nvidia-smi >/dev/null 2>&1; then
        echo "✅ nvidia-smi working immediately!"
    else
        echo "⚠️  nvidia-smi not working yet - may need reboot"
        echo "Attempting to restart nvidia services..."
        
        # Try to restart nvidia services
        sudo systemctl restart nvidia-persistenced 2>/dev/null || echo "nvidia-persistenced not available"
        sudo modprobe nvidia 2>/dev/null || echo "nvidia module not ready"
        
        # Test again
        sleep 2
        if nvidia-smi >/dev/null 2>&1; then
            echo "✅ nvidia-smi working after service restart!"
        else
            echo "❌ nvidia-smi still not working - proceeding anyway"
        fi
    fi
fi

# Navigate to transcription environment
cd /opt/transcribe
source venv/bin/activate

# Check current PyTorch CUDA status
echo ""
echo "🔧 Checking PyTorch CUDA status..."
python -c "
import torch
print(f'Current PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
else:
    print('CUDA not available - will reinstall PyTorch')
"

# Reinstall CUDA-enabled PyTorch if needed
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$CUDA_AVAILABLE" = "False" ]; then
    echo "🔧 Installing CUDA-enabled PyTorch..."
    
    # Uninstall current PyTorch
    pip uninstall -y torch torchvision torchaudio
    
    # Install CUDA-enabled PyTorch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    echo "✅ PyTorch installation complete"
    
    # Test again
    python -c "
import torch
print(f'New PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
else
    echo "✅ PyTorch CUDA already working"
fi

# Build model cache
echo ""
echo "🔧 Building model cache..."

# Check if cache already exists
if [ -f "/opt/transcribe/cache/cache_info.json" ]; then
    echo "✅ Cache already exists"
    cat /opt/transcribe/cache/cache_info.json
else
    echo "Creating model cache..."
    
    # Create cache directory
    mkdir -p /opt/transcribe/cache
    
    # Create a quick cache builder script
    cat > build_cache_quick.py << 'CACHE_SCRIPT'
import os
import torch
import json
import warnings
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

warnings.filterwarnings("ignore")

print("🧠 Quick Model Cache Building...")

# Configuration
model_id = "KBLab/kb-whisper-small"
cache_dir = "/opt/transcribe/models"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Device: {device} | Model: {model_id}")

# Enable optimizations if CUDA available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Download and cache model
print("⬇️ Downloading and caching model...")
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        cache_dir=cache_dir,
        device_map="auto" if device == "cuda" else None,
        local_files_only=False
    )

    print("⬇️ Caching processor...")
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    print("✅ Model and processor cached successfully!")
    
    # Test kernel compilation if CUDA
    kernels_compiled = False
    if device == "cuda":
        print("🛠️ Testing CUDA operations...")
        try:
            # Quick test
            test_tensor = torch.randn(100, 100, device=device)
            _ = torch.matmul(test_tensor, test_tensor)
            kernels_compiled = True
            print("✅ CUDA operations working")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ CUDA test failed: {e}")
    
    # Create cache info
    cache_info = {
        "model_id": model_id,
        "cache_dir": cache_dir,
        "device": device,
        "torch_dtype": str(torch_dtype),
        "kernels_compiled": kernels_compiled,
        "timestamp": datetime.now().isoformat()
    }

    # Save cache info
    os.makedirs("/opt/transcribe/cache", exist_ok=True)
    with open("/opt/transcribe/cache/cache_info.json", "w") as f:
        json.dump(cache_info, f, indent=2)

    print("✅ Cache info saved!")
    print(f"📍 Models cached in: {cache_dir}")
    print(f"📄 Cache info saved to: /opt/transcribe/cache/cache_info.json")
    print(f"🚀 Kernels compiled: {kernels_compiled}")
    
except Exception as e:
    print(f"❌ Cache building failed: {e}")
    import traceback
    traceback.print_exc()
CACHE_SCRIPT

    # Run cache builder
    python build_cache_quick.py
    
    # Clean up
    rm -f build_cache_quick.py
fi

echo ""
echo "🎯 CRITICAL FIXES INSTALLATION COMPLETE"
echo "========================================"
echo "Summary:"
nvidia-smi >/dev/null 2>&1 && echo "✅ NVIDIA drivers: Working" || echo "❌ NVIDIA drivers: Failed"
python -c "import torch; print('✅ PyTorch CUDA: Working' if torch.cuda.is_available() else '❌ PyTorch CUDA: Failed')" 2>/dev/null
[ -f "/opt/transcribe/cache/cache_info.json" ] && echo "✅ Model cache: Created" || echo "❌ Model cache: Failed"

INSTALL_FIXES_EOF

echo ""
echo "=================================================="
echo "🔍 RUNNING COMPREHENSIVE DIAGNOSTICS"
echo "=================================================="

# Run comprehensive diagnostics
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'DIAGNOSTIC_EOF'
set -e

echo "🔍 COMPREHENSIVE INSTANCE DIAGNOSTICS"
echo "======================================"

# Basic system info
echo "System Information:"
echo "  Instance type: $(curl -s --connect-timeout 3 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'Unknown')"
echo "  Instance ID: $(curl -s --connect-timeout 3 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo 'Unknown')"
echo "  Region: $(curl -s --connect-timeout 3 http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo 'Unknown')"
echo "  Zone: $(curl -s --connect-timeout 3 http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null || echo 'Unknown')"
echo "  Uptime: $(uptime)"

echo ""
echo "Quick System Checks:"

# Check GPU immediately
echo "GPU Status:"
if nvidia-smi >/dev/null 2>&1; then
    echo "  ✅ nvidia-smi working"
    nvidia-smi --query-gpu=name,memory.total,driver_version,temperature.gpu --format=csv,noheader
else
    echo "  ❌ nvidia-smi not working"
fi

# Check directories
echo ""
echo "Directory Structure:"
for dir in "/opt/transcribe" "/opt/transcribe/venv" "/opt/transcribe/models" "/opt/transcribe/cache" "/opt/transcribe/scripts"; do
    if [ -d "$dir" ]; then
        count=$(ls "$dir" 2>/dev/null | wc -l)
        echo "  ✅ $dir ($count items)"
    else
        echo "  ❌ $dir missing"
    fi
done

# Check key files
echo ""
echo "Key Files:"
for file in "/opt/transcribe/cache/cache_info.json" "/opt/transcribe/scripts/transcribe_optimized.py" "/opt/transcribe/scripts/api_server.py"; do
    if [ -f "$file" ]; then
        size=$(stat -c%s "$file" 2>/dev/null || echo "unknown")
        echo "  ✅ $file ($size bytes)"
    else
        echo "  ❌ $file missing"
    fi
done

# Check Python environment
echo ""
echo "Python Environment:"
if [ -d "/opt/transcribe" ]; then
    cd /opt/transcribe
    if [ -f "venv/bin/activate" ]; then
        echo "  ✅ Virtual environment found"
        source venv/bin/activate
        
        echo "  Python: $(python --version 2>&1)"
        
        # Check PyTorch
        python -c "
try:
    import torch
    print(f'  ✅ PyTorch {torch.__version__}')
    print(f'  CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU Count: {torch.cuda.device_count()}')
        print(f'  GPU 0: {torch.cuda.get_device_name(0)}')
        # Quick GPU test
        x = torch.randn(100, 100, device='cuda')
        print(f'  ✅ GPU tensor allocation successful')
        del x
        torch.cuda.empty_cache()
    else:
        print('  ⚠️  CUDA not available - will use CPU')
except Exception as e:
    print(f'  ❌ PyTorch error: {e}')
" 2>/dev/null || echo "  ❌ Python check failed"
    else
        echo "  ❌ Virtual environment not found"
    fi
else
    echo "  ❌ /opt/transcribe not found"
fi

# Run detailed diagnostic script
echo ""
echo "=================================================="
echo "RUNNING DETAILED DIAGNOSTIC SCRIPT"
echo "=================================================="

if [ -f "/tmp/diagnostic_check.py" ]; then
    cd /opt/transcribe 2>/dev/null && source venv/bin/activate 2>/dev/null
    python /tmp/diagnostic_check.py
else
    echo "❌ Detailed diagnostic script not available"
fi

echo ""
echo "=================================================="
echo "DIAGNOSTIC COMPLETE"
echo "=================================================="

DIAGNOSTIC_EOF

echo ""
echo "=================================================="
echo "🎯 DIAGNOSTIC SESSION COMPLETE"
echo "=================================================="

# Ask if user wants to keep instance running
echo ""
echo "Keep diagnostic instance running for further testing? (y/N): "
read -t 10 -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Diagnostic instance left running: $INSTANCE_ID"
    echo "Public IP: $PUBLIC_IP"
    echo "Connect with: ssh -i transcription-ec2.pem ubuntu@$PUBLIC_IP"
    echo "Or cleanup later with: aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
    
    # Clear trap to prevent auto-cleanup
    trap - EXIT
else
    echo "Auto-terminating diagnostic instance..."
    # Cleanup will be handled by trap
fi

echo ""
echo "🔍 Diagnostic session complete!" 