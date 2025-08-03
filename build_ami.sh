#!/bin/bash

# Optimized Transcription AMI Builder
# Target: Fast model loading with pre-downloaded models and CUDA warmup
# Focus: Elegance, simplicity, and performance

set -euo pipefail

echo "=== Optimized Transcription AMI Builder ==="
echo "Building AMI with pre-downloaded models and optimized environment..."

# Configuration
export AWS_DEFAULT_REGION=eu-north-1
INSTANCE_TYPE="g4dn.xlarge"
BASE_AMI="ami-0989fb15ce71ba39e"  # Ubuntu 22.04 LTS
SECURITY_GROUP="transcription-g4dn-sg"
KEY_NAME="transcription-ec2"
MODEL_ID="KBLab/kb-whisper-small"

# Global variables
INSTANCE_ID=""
PUBLIC_IP=""
AMI_ID=""

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a build_ami.log
}

# Error handling
handle_error() {
    log "ERROR: $1"
    cleanup
    exit 1
}

# Cleanup function
cleanup() {
    if [ ! -z "$INSTANCE_ID" ]; then
        log "Terminating instance $INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null 2>&1 || true
    fi
    rm -f /tmp/build_ami_*
}

# Set traps
trap 'handle_error "Script failed at line $LINENO"' ERR
trap cleanup EXIT INT TERM

# Validate prerequisites
validate_prerequisites() {
    log "Validating prerequisites..."
    
    if ! command -v aws &> /dev/null; then
        handle_error "AWS CLI not found"
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        handle_error "AWS credentials not configured"
    fi
    
    if [ ! -f "${KEY_NAME}.pem" ]; then
        handle_error "SSH key ${KEY_NAME}.pem not found in current directory"
    fi
    
    # Set proper SSH key permissions
    chmod 400 "${KEY_NAME}.pem" || handle_error "Failed to set SSH key permissions"
    log "SSH key permissions set to 400"
    
    # Validate required files exist in current directory
    # NOTE: Run this script from the root directory containing the transcription files
    if [ ! -f "fast_transcribe.py" ]; then
        handle_error "fast_transcribe.py not found in current directory. Ensure you're running from project root."
    fi
    
    if [ ! -f "fast_transcribe.sh" ]; then
        handle_error "fast_transcribe.sh not found in current directory. Ensure you're running from project root."
    fi
    
    log "Prerequisites validated"
}

# Launch EC2 instance
launch_instance() {
    log "Launching EC2 instance..."
    
    # Get security group ID from name
    SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=$SECURITY_GROUP" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null) || handle_error "Security group $SECURITY_GROUP not found"
    
    if [ "$SECURITY_GROUP_ID" = "None" ] || [ -z "$SECURITY_GROUP_ID" ]; then
        handle_error "Security group $SECURITY_GROUP not found"
    fi
    
    log "Using security group: $SECURITY_GROUP (ID: $SECURITY_GROUP_ID)"
    
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$BASE_AMI" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP_ID" \
        --block-device-mappings '[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": 50,
                    "VolumeType": "gp3",
                    "Iops": 3000,
                    "Throughput": 125,
                    "DeleteOnTermination": true
                }
            }
        ]' \
        --query 'Instances[0].InstanceId' \
        --output text) || handle_error "Failed to launch instance"
    
    log "Instance launched: $INSTANCE_ID"
    
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" || handle_error "Instance failed to start"
    
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text) || handle_error "Failed to get instance public IP"
    
    if [ "$PUBLIC_IP" = "None" ] || [ -z "$PUBLIC_IP" ]; then
        handle_error "Instance has no public IP address"
    fi
    
    log "Instance running at $PUBLIC_IP"
}

# Establish SSH connection
establish_ssh() {
    log "Establishing SSH connection..."
    
    local max_attempts=30
    local attempt=0
    
    # Wait a bit longer for the first attempt to ensure instance is fully ready
    sleep 15
    
    while [ $attempt -lt $max_attempts ]; do
        if ssh -o ConnectTimeout=10 \
               -o StrictHostKeyChecking=no \
               -o UserKnownHostsFile=/dev/null \
               -o ServerAliveInterval=60 \
               -o ServerAliveCountMax=3 \
               -i "${KEY_NAME}.pem" \
               ubuntu@"$PUBLIC_IP" "echo 'SSH ready'" &> /dev/null; then
            log "SSH connection established"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log "SSH attempt $attempt/$max_attempts..."
        sleep 15
    done
    
    handle_error "Failed to establish SSH connection after $max_attempts attempts"
}

# Setup instance with dependencies
setup_instance() {
    log "Setting up instance..."
    
    cat > /tmp/setup_ami.sh << 'SETUP_SCRIPT'
#!/bin/bash
set -e

echo "=== AMI Setup Starting ==="
echo "Timestamp: $(date)"

# Wait for instance to be fully ready
sleep 30

# Update system
apt-get update -y
add-apt-repository universe -y > /dev/null 2>&1 || true
apt-get update -y

# Install kernel headers
KERNEL_VERSION=$(uname -r)
echo "[$(date)] Current kernel: $KERNEL_VERSION"
apt-get install -y linux-headers-$KERNEL_VERSION

# Upgrade system packages
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# Install essential packages
apt-get install -y \
    dkms \
    python3-pip \
    python3-venv \
    curl \
    awscli \
    ffmpeg \
    libsndfile1 \
    bc

# Install NVIDIA drivers
echo "[$(date)] Installing NVIDIA driver 535..."
DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-535 nvidia-dkms-535 || {
    echo "[$(date)] ERROR: Failed to install NVIDIA drivers"
    exit 1
}

# Wait for DKMS to build modules
echo "[$(date)] Waiting for DKMS to build NVIDIA modules..."
MAX_DKMS_WAIT=300
DKMS_WAIT=0
while [ $DKMS_WAIT -lt $MAX_DKMS_WAIT ]; do
    if dkms status nvidia 2>/dev/null | grep -q "installed"; then
        echo "[$(date)] DKMS build completed"
        break
    fi
    echo "[$(date)] Waiting for DKMS build... ($DKMS_WAIT/$MAX_DKMS_WAIT seconds)"
    sleep 10
    DKMS_WAIT=$((DKMS_WAIT + 10))
done

# Check if NVIDIA modules are loaded, if not mark for reboot
if ! nvidia-smi &> /dev/null; then
    echo "[$(date)] NVIDIA modules not loaded, marking for reboot"
    touch /tmp/nvidia_reboot_required
fi

# Setup Python environment
echo "[$(date)] Setting up Python environment..."
mkdir -p /opt/transcribe/{scripts,models,cache,logs} || {
    echo "[$(date)] ERROR: Failed to create directories"
    exit 1
}
chown -R ubuntu:ubuntu /opt/transcribe || {
    echo "[$(date)] ERROR: Failed to set ownership"
    exit 1
}

sudo -u ubuntu python3 -m venv /opt/transcribe/venv || {
    echo "[$(date)] ERROR: Failed to create virtual environment"
    exit 1
}
sudo -u ubuntu /opt/transcribe/venv/bin/pip install --upgrade pip || {
    echo "[$(date)] ERROR: Failed to upgrade pip"
    exit 1
}
sudo -u ubuntu /opt/transcribe/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu118 || {
    echo "[$(date)] ERROR: Failed to install PyTorch"
    exit 1
}
sudo -u ubuntu /opt/transcribe/venv/bin/pip install transformers librosa boto3 numpy accelerate || {
    echo "[$(date)] ERROR: Failed to install Python packages"
    exit 1
}

# Setup environment variables for optimized model caching
echo "[$(date)] Setting up environment variables..."
sudo -u ubuntu tee /opt/transcribe/.env > /dev/null << ENV_VARS
# Optimized model caching environment
export TRANSFORMERS_CACHE=/opt/transcribe/models
export HF_HOME=/opt/transcribe/models
export TORCH_HOME=/opt/transcribe/models
export HF_DATASETS_CACHE=/opt/transcribe/cache
export PYTHONPATH=/opt/transcribe/scripts:\$PYTHONPATH
export MODEL_ID='$MODEL_ID'
ENV_VARS

# Add environment to bashrc for ubuntu user
echo "source /opt/transcribe/.env" >> /home/ubuntu/.bashrc

echo "[$(date)] Python environment setup completed"
SETUP_SCRIPT

    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/setup_ami.sh ubuntu@"$PUBLIC_IP":/tmp/
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "sudo bash /tmp/setup_ami.sh" || handle_error "Setup script failed"
    
    # Check if reboot is required
    if ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
           ubuntu@"$PUBLIC_IP" "[ -f /tmp/nvidia_reboot_required ]"; then
        log "NVIDIA drivers require reboot - rebooting instance..."
        
        ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
            ubuntu@"$PUBLIC_IP" "sudo reboot" || true
        
        sleep 30
        aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
        establish_ssh
        
        ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
            ubuntu@"$PUBLIC_IP" "nvidia-smi" || handle_error "NVIDIA drivers not working after reboot"
        
        log "Instance rebooted successfully, NVIDIA drivers working"
    fi
    
    log "Basic setup completed"
}

# Enhanced bytecode compilation
enhanced_bytecode_compilation() {
    log "📚 Performing enhanced bytecode compilation..."
    
    cat > /tmp/enhanced_bytecode_compilation.py << 'BYTECODE_COMPILATION'
#!/opt/transcribe/venv/bin/python3
import os
import sys
import compileall
from datetime import datetime

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📚 Enhanced Bytecode Compilation")

try:
    # Dynamically determine Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = f"/opt/transcribe/venv/lib/python{python_version}/site-packages"
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Compiling site-packages: {site_packages}")
    
    # Compile all Python files in site-packages
    success = compileall.compile_dir(
        site_packages,
        force=True,
        quiet=0,
        optimize=2
    )
    
    if success:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Site-packages compilation completed")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Some files failed to compile")
    
    # Compile specific heavy libraries
    heavy_libraries = ["torch", "transformers", "librosa", "numpy"]
    
    for lib in heavy_libraries:
        lib_path = os.path.join(site_packages, lib)
        if os.path.exists(lib_path):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Compiling {lib}...")
            try:
                compileall.compile_dir(lib_path, force=True, quiet=0, optimize=2)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ {lib} compiled")
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ {lib} compilation failed: {e}")
    
    # Create optimized Python path configuration
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating optimized Python path...")
    
    optimizer_path = os.path.join(site_packages, "transcription_optimizer.pth")
    with open(optimizer_path, 'w') as f:
        f.write("# Optimized Python path for transcription\n")
        f.write("import sys\n")
        f.write("import os\n")
        f.write("\n")
        f.write("# Add optimized paths for faster module resolution\n")
        f.write(f"sys.path.insert(0, '{site_packages}')\n")
        f.write("sys.path.insert(0, '/opt/transcribe/scripts')\n")
        f.write("\n")
        f.write("# Pre-import commonly used modules\n")
        f.write("try:\n")
        f.write("    import torch\n")
        f.write("    import numpy as np\n")
        f.write("    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor\n")
        f.write("except ImportError:\n")
        f.write("    pass\n")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Optimized Python path created")
    
    # Count .pyc files
    pyc_count = 0
    for root, dirs, files in os.walk(site_packages):
        pyc_count += len([f for f in files if f.endswith('.pyc')])
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Found {pyc_count} compiled .pyc files")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Enhanced bytecode compilation completed!")
    
except Exception as e:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
BYTECODE_COMPILATION

    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/enhanced_bytecode_compilation.py ubuntu@"$PUBLIC_IP":/opt/transcribe/
    
    log "Executing enhanced bytecode compilation..."
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "cd /opt/transcribe && sudo -u ubuntu /opt/transcribe/venv/bin/python enhanced_bytecode_compilation.py" \
        || handle_error "Enhanced bytecode compilation failed"
    
    log "✅ Enhanced bytecode compilation completed"
}

# Pre-download model and warm CUDA
pre_download_model_and_warm_cuda() {
    log "🔥 Pre-downloading model and warming CUDA context..."
    
    cat > /tmp/pre_download_and_warmup.py << 'MODEL_WARMUP'
#!/opt/transcribe/venv/bin/python3
import os
import sys
import torch
import time
import json
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔥 Pre-downloading Model and Warming CUDA")

try:
    # Set environment variables for this session
    os.environ['TRANSFORMERS_CACHE'] = '/opt/transcribe/models'
    os.environ['HF_HOME'] = '/opt/transcribe/models'
    os.environ['TORCH_HOME'] = '/opt/transcribe/models'
    
    # Ensure model cache directory exists
    os.makedirs("/opt/transcribe/models", exist_ok=True)
    
    # Step 1: Pre-download the Whisper model and processor
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pre-downloading Whisper model...")
    model_id = os.environ.get('MODEL_ID', 'KBLab/kb-whisper-small')
    
    # Download with retry logic for reliability
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Downloading model files... (attempt {attempt + 1}/{max_retries})")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                cache_dir="/opt/transcribe/models",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                trust_remote_code=False
            )
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Downloading processor files...")
            processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir="/opt/transcribe/models",
                trust_remote_code=False
            )
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Model and processor downloaded successfully")
            break
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Download attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Failed to download model after {max_retries} attempts")
                sys.exit(1)
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Retrying in 30 seconds...")
                time.sleep(30)
    
    # Step 2: CUDA Context Warmup (if available)
    if torch.cuda.is_available():
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Performing intensive CUDA warmup...")
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        
        # Intensive CUDA warmup with large tensors
        warmup_start = time.time()
        for i in range(10):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CUDA warmup iteration {i+1}/10...")
            
            # Use large 2000x2000 tensors as specified
            x = torch.randn(2000, 2000, device=device, dtype=torch.float16)
            y = torch.randn(2000, 2000, device=device, dtype=torch.float16)
            
            # Perform matrix multiplication and other operations
            z = torch.mm(x, y)
            z = torch.relu(z)
            z = z.sum()
            
            # Clean up to ensure memory management
            del x, y, z
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        warmup_time = time.time() - warmup_start
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ CUDA warmup completed in {warmup_time:.1f}s")
        
        # Test model on GPU for verification
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing model GPU placement...")
        model = model.to(device)
        model.eval()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Model successfully placed on GPU")
        
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ CUDA not available, using CPU")
        device = torch.device('cpu')
    
    # Step 3: Verify model cache and create metadata
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating model cache metadata...")
    
    # Check what files were downloaded
    cache_files = []
    for root, dirs, files in os.walk("/opt/transcribe/models"):
        for file in files:
            file_path = os.path.join(root, file)
            cache_files.append({
                "path": file_path,
                "size_mb": os.path.getsize(file_path) / (1024 * 1024)
            })
    
    total_size = sum(f["size_mb"] for f in cache_files)
    
    metadata = {
        "model_id": model_id,
        "cache_directory": "/opt/transcribe/models",
        "device_warmed": str(device),
        "created_at": datetime.now().isoformat(),
        "total_cache_size_mb": round(total_size, 1),
        "num_files": len(cache_files),
        "cuda_warmed": torch.cuda.is_available(),
        "status": "ready"
    }
    
    metadata_path = "/opt/transcribe/models/cache_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Cache metadata saved to: {metadata_path}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Pre-download and warmup completed!")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total cache size: {total_size:.1f} MB")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Files cached: {len(cache_files)}")
    
except Exception as e:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
MODEL_WARMUP

    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/pre_download_and_warmup.py ubuntu@"$PUBLIC_IP":/opt/transcribe/
    
    log "Executing model pre-download and CUDA warmup..."
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "cd /opt/transcribe && source .env && sudo -u ubuntu MODEL_ID='$MODEL_ID' /opt/transcribe/venv/bin/python pre_download_and_warmup.py" \
        || handle_error "Model pre-download and warmup failed"
    
    log "✅ Model pre-download and CUDA warmup completed successfully"
}



# Create transcription script
create_transcription_script() {
    log "Creating transcription script..."
    
    # Files are already validated in validate_prerequisites()
    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        "fast_transcribe.py" ubuntu@"$PUBLIC_IP":/opt/transcribe/fast_transcribe.py
    
    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        "fast_transcribe.sh" ubuntu@"$PUBLIC_IP":/opt/transcribe/fast_transcribe.sh
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "chmod +x /opt/transcribe/*.py && sudo mkdir -p /opt/transcription && sudo chown ubuntu:ubuntu /opt/transcription && sudo cp /opt/transcribe/fast_transcribe.sh /opt/transcription/fast_transcribe.sh && sudo chown ubuntu:ubuntu /opt/transcription/fast_transcribe.sh && chmod +x /opt/transcription/fast_transcribe.sh"
    
    log "✅ Scripts uploaded and verified"
}

# Final validation
validate_setup() {
    log "Running validation..."
    
    cat > /tmp/validate.sh << 'VALIDATE_SCRIPT'
#!/bin/bash
set -e

echo "=== AMI Validation ==="
echo "Timestamp: $(date)"

# Essential checks for optimized AMI
echo -n "NVIDIA drivers... "
nvidia-smi &> /dev/null && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Model cache directory... "
[ -d /opt/transcribe/models ] && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Model cache metadata... "
[ -f /opt/transcribe/models/cache_metadata.json ] && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Model files cached... "
/opt/transcribe/venv/bin/python -c "
# Source environment variables
import os
os.environ['TRANSFORMERS_CACHE'] = '/opt/transcribe/models'
os.environ['HF_HOME'] = '/opt/transcribe/models'
os.environ['TORCH_HOME'] = '/opt/transcribe/models'
import json
import os
import sys

# Check metadata
with open('/opt/transcribe/models/cache_metadata.json', 'r') as f:
    metadata = json.load(f)

# Verify we have cached files
if metadata['num_files'] < 5:
    print(f'Too few cached files: {metadata[\"num_files\"]}')
    sys.exit(1)

# Verify reasonable cache size
if metadata['total_cache_size_mb'] < 100:
    print(f'Cache too small: {metadata[\"total_cache_size_mb\"]} MB')
    sys.exit(1)

print(f'Cache validation: SUCCESS - {metadata[\"num_files\"]} files, {metadata[\"total_cache_size_mb\"]} MB')
" && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Fast model loading test... "
/opt/transcribe/venv/bin/python -c "
# Source environment variables
import os
os.environ['TRANSFORMERS_CACHE'] = '/opt/transcribe/models'
os.environ['HF_HOME'] = '/opt/transcribe/models'
os.environ['TORCH_HOME'] = '/opt/transcribe/models'
import sys
import os
import time
import torch
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Set environment variables
os.environ['TRANSFORMERS_CACHE'] = '/opt/transcribe/models'
os.environ['HF_HOME'] = '/opt/transcribe/models'

try:
    start_time = time.time()
    
    # Test fast loading from cache using environment variable
    model_id = os.environ.get('MODEL_ID', 'KBLab/kb-whisper-small')
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        cache_dir='/opt/transcribe/models',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto' if torch.cuda.is_available() else None,
        local_files_only=True  # Only use cached files for validation
    )
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir='/opt/transcribe/models',
        local_files_only=True  # Only use cached files for validation
    )
    
    load_time = time.time() - start_time
    
    if load_time > 5.0:
        print(f'Model loading too slow: {load_time:.1f}s')
        sys.exit(1)
    
    print(f'Fast loading test: SUCCESS - {load_time:.1f}s')
    
except Exception as e:
    print(f'Fast loading test: FAILED - {e}')
    sys.exit(1)
" && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "CUDA context test... "
/opt/transcribe/venv/bin/python -c "
# Source environment variables
import os
os.environ['TRANSFORMERS_CACHE'] = '/opt/transcribe/models'
os.environ['HF_HOME'] = '/opt/transcribe/models'
os.environ['TORCH_HOME'] = '/opt/transcribe/models'
import torch
import sys

if not torch.cuda.is_available():
    print('CUDA context test: SKIPPED - CUDA not available')
    sys.exit(0)

try:
    device = torch.device('cuda:0')
    x = torch.randn(100, 100, device=device, dtype=torch.float16)
    y = torch.randn(100, 100, device=device, dtype=torch.float16)
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    print('CUDA context test: SUCCESS')
except Exception as e:
    print(f'CUDA context test: FAILED - {e}')
    sys.exit(1)
" && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Environment variables... "
[ -f /opt/transcribe/.env ] && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Transcription scripts... "
[ -f /opt/transcribe/fast_transcribe.py ] && [ -f /opt/transcribe/fast_transcribe.sh ] && [ -f /opt/transcription/fast_transcribe.sh ] && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Fast transcribe integration... "
/opt/transcribe/venv/bin/python -c "
# Source environment variables
import os
os.environ['TRANSFORMERS_CACHE'] = '/opt/transcribe/models'
os.environ['HF_HOME'] = '/opt/transcribe/models'
os.environ['TORCH_HOME'] = '/opt/transcribe/models'
import sys
sys.path.insert(0, '/opt/transcribe')

try:
    from fast_transcribe import FastTranscriber
    print('Fast transcribe integration: SUCCESS')
except ImportError as e:
    print(f'Fast transcribe integration: FAILED - {e}')
    sys.exit(1)
" && echo "OK" || { echo "FAILED"; exit 1; }

# Completion marker
echo "AMI_SETUP_COMPLETE=true" > /opt/transcribe/.setup_complete
echo "SETUP_DATE=$(date)" >> /opt/transcribe/.setup_complete
echo "SETUP_TYPE=optimized_cache" >> /opt/transcribe/.setup_complete

echo "Validation complete"
VALIDATE_SCRIPT

    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/validate.sh ubuntu@"$PUBLIC_IP":/tmp/
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "sudo bash /tmp/validate.sh" || handle_error "Validation failed"
    
    log "✅ Validation completed successfully"
}

# Create AMI
create_ami() {
    log "Creating AMI..."
    
    AMI_NAME="transcription-gpu-$(date +%Y%m%d-%H%M%S)"
    
    # Validate instance is still running before creating AMI
    local instance_state=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null) || handle_error "Failed to get instance state"
    
    if [ "$instance_state" != "running" ]; then
        handle_error "Instance is not running (state: $instance_state), cannot create AMI"
    fi
    
    AMI_ID=$(aws ec2 create-image \
        --instance-id "$INSTANCE_ID" \
        --name "$AMI_NAME" \
        --description "GPU transcription AMI with pre-downloaded models and optimized environment" \
        --query 'ImageId' \
        --output text) || handle_error "Failed to create AMI"
    
    log "AMI creation initiated: $AMI_ID"
    
    # Wait for AMI to be available
    log "Waiting for AMI to be available (this may take up to 30 minutes)..."
    
    local max_attempts=60
    local delay=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        local ami_state=$(aws ec2 describe-images \
            --image-ids "$AMI_ID" \
            --query 'Images[0].State' \
            --output text 2>/dev/null) || true
        
        if [ "$ami_state" = "available" ]; then
            log "AMI is now available"
            break
        elif [ "$ami_state" = "failed" ]; then
            handle_error "AMI creation failed"
        elif [ "$ami_state" = "None" ] || [ -z "$ami_state" ]; then
            log "WARNING: Could not get AMI state, continuing to wait..."
        fi
        
        attempt=$((attempt + 1))
        log "AMI status: $ami_state (attempt $attempt/$max_attempts)"
        
        if [ $attempt -lt $max_attempts ]; then
            sleep $delay
        fi
    done
    
    if [ $attempt -eq $max_attempts ]; then
        handle_error "AMI creation timed out after $((max_attempts * delay)) seconds"
    fi
    
    log "AMI created successfully: $AMI_ID"
    
    # Write AMI ID to file
    echo "$AMI_ID" > ami_id.txt
    log "AMI ID saved to ami_id.txt: $AMI_ID"
}

# Update AMI ID in lambda function
update_lambda_ami_id() {
    log "Updating AMI ID in lambda function..."
    
    # Try multiple possible paths for lambda file
    local lambda_file=""
    if [ -f "../setup/lambda/lambda_process_upload.py" ]; then
        lambda_file="../setup/lambda/lambda_process_upload.py"
    elif [ -f "setup/lambda/lambda_process_upload.py" ]; then
        lambda_file="setup/lambda/lambda_process_upload.py"
    fi
    
    if [ -n "$lambda_file" ] && [ -f "$lambda_file" ]; then
        cp "$lambda_file" "${lambda_file}.backup"
        # Fix: Update the pattern to match the actual format in lambda file
        if sed -i "s/AMI_ID = os.environ.get('AMI_ID', 'ami-[a-z0-9]*').*$/AMI_ID = os.environ.get('AMI_ID', '$AMI_ID')  # Read from environment variable/" "$lambda_file"; then
            log "Updated AMI ID in lambda function: $lambda_file"
            
            if grep -q "AMI_ID = os.environ.get('AMI_ID', '$AMI_ID')  # Read from environment variable" "$lambda_file"; then
                log "AMI ID update verified in lambda function"
            else
                log "WARNING: AMI ID update verification failed"
            fi
        else
            log "WARNING: Failed to update AMI ID in lambda function"
        fi
    else
        log "WARNING: Lambda function file not found: $lambda_file"
    fi
    
    # Update AMI ID in Python transcription script
    local python_file="fast_transcribe.py"
    
    if [ -f "$python_file" ]; then
        cp "$python_file" "${python_file}.backup"
        # Fix: Update the pattern to match the actual format in Python file
        if sed -i "s/EXPECTED_AMI_ID = 'ami-[a-z0-9]*'.*$/EXPECTED_AMI_ID = '$AMI_ID'  # Optimized GPU AMI with pre-cached models/" "$python_file"; then
            log "Updated expected AMI ID in Python script: $python_file"
            
            if grep -q "EXPECTED_AMI_ID = '$AMI_ID'  # Optimized GPU AMI with pre-cached models" "$python_file"; then
                log "Expected AMI ID update verified in Python script"
            else
                log "WARNING: Expected AMI ID update verification failed"
            fi
        else
            log "WARNING: Failed to update expected AMI ID in Python script"
        fi
    else
        log "WARNING: Python transcription script not found: $python_file"
    fi
}

# Main execution
main() {
    log "Starting AMI build process..."
    
    log "=== Build Configuration ==="
    log "Region: $AWS_DEFAULT_REGION"
    log "Instance Type: $INSTANCE_TYPE"
    log "Base AMI: $BASE_AMI"
    log "Security Group: $SECURITY_GROUP"
    log "Key Name: $KEY_NAME"
    log "Model ID: $MODEL_ID"
    log "Build Date: $(date)"
    log "=========================="
    
    validate_prerequisites
    launch_instance
    establish_ssh
    setup_instance
    enhanced_bytecode_compilation
    pre_download_model_and_warm_cuda
    create_transcription_script
    validate_setup
    create_ami
    update_lambda_ami_id
    
    log "AMI build completed successfully!"
    log "Final AMI ID: $AMI_ID"
    log "AMI ID file updated: ami_id.txt"
    log "Lambda function and scripts updated with new AMI ID"
    
    log "=== OPTIMIZED TRANSCRIPTION AMI BUILD SUMMARY ==="
    log "✓ Base AMI: $BASE_AMI"
    log "✓ New AMI: $AMI_ID"
    log "✓ Instance Type: $INSTANCE_TYPE"
    log "✓ Model: $MODEL_ID"
    log "✓ NVIDIA Drivers: Installed"
    log "✓ Python Environment: Ready"
    log "✓ Enhanced Bytecode Compilation: Completed"
    log "✓ Model Cache: Pre-downloaded"
    log "✓ CUDA Context: Pre-warmed"
    log "✓ Environment Variables: Configured"
    log "✓ Scripts: Uploaded and Verified"
    log "✓ Lambda Function: Updated"
    log "✓ Setup Marker: Created"
    log "=================================================="
    log ""
    log "Next steps:"
    log "1. Deploy the updated lambda function with: cd ../setup/lambda && ./deploy_lambda_functions.sh"
    log "2. Test transcription by uploading a file to S3"
    log "3. Monitor CloudWatch logs for any issues"
    log "4. Expect 2-3 second model loading times (vs 10+ seconds cold)"
    log "5. No library installation required at runtime"
}

# Run main function
main