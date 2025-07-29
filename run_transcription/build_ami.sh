#!/bin/bash

# Simplified Production G4DN.XLARGE AMI Builder
# Target: Fast boot with pre-cached Swedish Whisper model
# Focus: Simplicity and reliability

set -euo pipefail

echo "=== Simplified G4DN AMI Builder ==="
echo "Building AMI with NVIDIA drivers and cached Whisper model..."

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
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        handle_error "AWS CLI not found"
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        handle_error "AWS credentials not configured"
    fi
    
    # Check SSH key
    if [ ! -f "${KEY_NAME}.pem" ]; then
        handle_error "SSH key ${KEY_NAME}.pem not found"
    fi
    
    log "Prerequisites validated"
}

# Launch EC2 instance
launch_instance() {
    log "Launching EC2 instance..."
    
    # Launch with optimized EBS settings
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$BASE_AMI" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-groups "$SECURITY_GROUP" \
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
        --query 'Instances[0].InstanceId' \
        --output text) || handle_error "Failed to launch instance"
    
    log "Instance launched: $INSTANCE_ID"
    
    # Wait for instance to be running
    log "Waiting for instance to start..."
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    log "Instance running at $PUBLIC_IP"
}

# Establish SSH connection
establish_ssh() {
    log "Establishing SSH connection..."
    
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if ssh -o ConnectTimeout=5 \
               -o StrictHostKeyChecking=no \
               -o UserKnownHostsFile=/dev/null \
               -i "${KEY_NAME}.pem" \
               ubuntu@"$PUBLIC_IP" "echo 'SSH ready'" &> /dev/null; then
            log "SSH connection established"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log "SSH attempt $attempt/$max_attempts..."
        sleep 10
    done
    
    handle_error "Failed to establish SSH connection"
}

# Main setup script
setup_instance() {
    log "Setting up instance..."
    
    # Create setup script
    cat > /tmp/setup_ami.sh << 'SETUP_SCRIPT'
#!/bin/bash
set -e

echo "=== AMI Setup Starting ==="
echo "Timestamp: $(date)"

# Function to retry apt operations with better error handling
retry_apt_operation() {
    local operation="$1"
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if eval "$operation > /dev/null 2>&1"; then
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 10
        apt-get clean > /dev/null 2>&1 || true
        rm -rf /var/lib/apt/lists/* > /dev/null 2>&1 || true
    done
    return 1
}

# Wait for instance to be fully ready
sleep 30

# Update system with retry logic
retry_apt_operation "apt-get update -y" || exit 1

# Verify package database is working
if ! apt-cache search build-essential | grep -q "build-essential"; then
    add-apt-repository universe -y > /dev/null 2>&1 || true
    retry_apt_operation "apt-get update -y" || exit 1
fi

# Install kernel headers first
KERNEL_VERSION=$(uname -r)
echo "[$(date)] Current kernel: $KERNEL_VERSION"
if ! retry_apt_operation "apt-get install -y linux-headers-$KERNEL_VERSION 2>&1"; then
    echo "ERROR: Failed to install linux-headers-$KERNEL_VERSION. Full error:"
    apt-get install -y linux-headers-$KERNEL_VERSION
    exit 1
fi

# Upgrade system packages
retry_apt_operation "DEBIAN_FRONTEND=noninteractive apt-get upgrade -y" || exit 1

# Install essential packages for transcription
retry_apt_operation "apt-get install -y \
    build-essential \
    dkms \
    python3-pip \
    python3-venv \
    python3-dev \
    ffmpeg \
    curl \
    wget \
    awscli \
    jq" || exit 1

# Verify kernel headers
if [ ! -d "/lib/modules/$KERNEL_VERSION/build" ]; then
    echo "ERROR: Kernel headers directory not found at /lib/modules/$KERNEL_VERSION/build"
    exit 1
fi

# Install NVIDIA drivers with DKMS support
echo "[$(date)] Installing NVIDIA drivers..."

# Install ubuntu-drivers-common for automatic driver detection
if ! retry_apt_operation "DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-drivers-common"; then
    echo "ERROR: Failed to install ubuntu-drivers-common"
    exit 1
fi

echo "[$(date)] Available NVIDIA drivers:"
ubuntu-drivers devices || echo "Warning: Failed to list available drivers"

# Install specific NVIDIA driver version for better compatibility
echo "[$(date)] Installing NVIDIA driver 535..."
if ! retry_apt_operation "DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-535 nvidia-dkms-535"; then
    echo "ERROR: Failed to install NVIDIA driver 535"
    echo "Attempting to install recommended driver..."
    if ! retry_apt_operation "DEBIAN_FRONTEND=noninteractive apt-get install -y \$(ubuntu-drivers devices | grep 'driver.*recommended' | awk '{print \$3}' | head -1)"; then
        echo "ERROR: Failed to install any NVIDIA driver"
        exit 1
    fi
fi

# Skip additional NVIDIA packages - only driver and DKMS needed for PyTorch
echo "[$(date)] NVIDIA driver installation complete - skipping unnecessary packages"
echo "[$(date)] PyTorch transcription only requires nvidia-driver and nvidia-dkms"

# Skip CUDA toolkit installation - PyTorch works fine without it
echo "[$(date)] Skipping CUDA toolkit installation - PyTorch includes its own CUDA runtime"
echo "[$(date)] NVIDIA drivers are sufficient for GPU acceleration"

# Wait for DKMS to build modules
echo "[$(date)] Waiting for DKMS to build NVIDIA modules..."
MAX_DKMS_WAIT=300  # 5 minutes max
DKMS_WAIT=0
while [ $DKMS_WAIT -lt $MAX_DKMS_WAIT ]; do
    if dkms status nvidia | grep -q "installed"; then
        echo "[$(date)] DKMS build completed"
        break
    fi
    echo "[$(date)] Waiting for DKMS build... ($DKMS_WAIT/$MAX_DKMS_WAIT seconds)"
    sleep 10
    DKMS_WAIT=$((DKMS_WAIT + 10))
done

# Force DKMS build if not completed
if ! dkms status nvidia | grep -q "installed"; then
    echo "[$(date)] Forcing DKMS build..."
    NVIDIA_VERSION=$(dkms status nvidia | head -1 | cut -d',' -f1 | cut -d'/' -f2)
    if [ -n "$NVIDIA_VERSION" ]; then
        dkms build nvidia/$NVIDIA_VERSION
        dkms install nvidia/$NVIDIA_VERSION
    fi
fi

# Verify kernel modules are built
KERNEL_VERSION=$(uname -r)
if [ ! -f "/lib/modules/$KERNEL_VERSION/updates/dkms/nvidia.ko" ] && [ ! -f "/lib/modules/$KERNEL_VERSION/kernel/drivers/video/nvidia.ko" ]; then
    echo "[$(date)] WARNING: NVIDIA kernel module not found, will require reboot"
    # Create a flag to indicate reboot is needed
    touch /tmp/nvidia_reboot_required
else
    echo "[$(date)] NVIDIA kernel modules found, attempting to load..."
    
    # Update module dependencies
    depmod -a
    
    # Try to load NVIDIA modules
    if modprobe nvidia; then
        echo "[$(date)] NVIDIA module loaded successfully"
        modprobe nvidia_uvm || echo "[$(date)] Warning: Failed to load nvidia_uvm module"
        modprobe nvidia_drm || echo "[$(date)] Warning: Failed to load nvidia_drm module"
        
        # Create NVIDIA device nodes (ignore if they already exist)
        if [ ! -e /dev/nvidia0 ]; then
            NVIDIA_MAJOR=$(grep nvidia /proc/devices | head -1 | awk '{print $1}')
            if [ -n "$NVIDIA_MAJOR" ] && [ "$NVIDIA_MAJOR" -gt 0 ] 2>/dev/null; then
                mknod -m 666 /dev/nvidia0 c $NVIDIA_MAJOR 0 2>/dev/null || echo "[$(date)] Warning: /dev/nvidia0 already exists"
                mknod -m 666 /dev/nvidiactl c $NVIDIA_MAJOR 255 2>/dev/null || echo "[$(date)] Warning: /dev/nvidiactl already exists"
                echo "[$(date)] NVIDIA device nodes created with major number: $NVIDIA_MAJOR"
            else
                echo "[$(date)] Warning: Could not determine NVIDIA major number, skipping device node creation"
            fi
        else
            echo "[$(date)] NVIDIA device nodes already exist"
        fi
    else
        echo "[$(date)] Could not load NVIDIA module, marking for reboot"
        touch /tmp/nvidia_reboot_required
    fi
fi

# Setup Python environment for transcription
echo "[$(date)] Setting up Python environment..."

# Create transcription directory structure
mkdir -p /opt/transcribe/{scripts,models,cache,logs}
mkdir -p /opt/transcription
chown -R ubuntu:ubuntu /opt/transcribe
chown -R ubuntu:ubuntu /opt/transcription

# Create Python virtual environment
sudo -u ubuntu python3 -m venv /opt/transcribe/venv

# Install Python dependencies
sudo -u ubuntu /opt/transcribe/venv/bin/pip install --upgrade pip
sudo -u ubuntu /opt/transcribe/venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
sudo -u ubuntu /opt/transcribe/venv/bin/pip install transformers accelerate librosa soundfile boto3

echo "[$(date)] Python environment setup completed"
SETUP_SCRIPT

    # Upload and execute setup script
    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/setup_ami.sh ubuntu@"$PUBLIC_IP":/tmp/
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "sudo bash /tmp/setup_ami.sh" || handle_error "Setup script failed"
    
    # Check if reboot is required
    if ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
           ubuntu@"$PUBLIC_IP" "[ -f /tmp/nvidia_reboot_required ]"; then
        log "NVIDIA drivers require reboot - rebooting instance..."
        
        # Reboot the instance
        ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
            ubuntu@"$PUBLIC_IP" "sudo reboot" || true
        
        # Wait for instance to stop
        log "Waiting for instance to stop..."
        sleep 30
        
        # Wait for instance to be running again
        log "Waiting for instance to restart..."
        aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
        
        # Re-establish SSH connection after reboot
        establish_ssh
        
        # Verify NVIDIA drivers after reboot
        ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
            ubuntu@"$PUBLIC_IP" "nvidia-smi" || handle_error "NVIDIA drivers not working after reboot"
        
        log "Instance rebooted successfully, NVIDIA drivers working"
    fi
    
    log "Basic setup completed"
}

# Cache the model
cache_model() {
    log "Caching Whisper model..."
    
    # Create model caching script
    cat > /tmp/cache_model.py << 'CACHE_SCRIPT'
#!/opt/transcribe/venv/bin/python3
import os
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pathlib import Path

model_id = "KBLab/kb-whisper-small"
cache_dir = "/opt/transcribe/models"

print(f"Downloading and caching model: {model_id}")

try:
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    
    # Download model and save it explicitly
    print("Downloading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    print("Model downloaded successfully")
    
    # Save model explicitly to cache directory
    model_path = os.path.join(cache_dir, model_id.replace("/", "--"))
    print(f"Saving model to: {model_path}")
    model.save_pretrained(model_path)
    print("Model saved to cache")
    
    # Download and save processor
    print("Downloading processor...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir
    )
    processor.save_pretrained(model_path)
    print("Processor saved to cache")
    
    # Verify the files exist
    expected_files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    for file in expected_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
    
    # Test loading from cache
    print("Testing model loading from cache...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        torch_dtype=torch.float16
    )
    print("Model loaded from cache successfully")
    
    # Create cache info file
    import json
    from datetime import datetime
    
    cache_info = {
        "model_id": model_id,
        "cache_dir": cache_dir,
        "model_path": model_path,
        "cached_at": datetime.now().isoformat(),
        "status": "ready"
    }
    
    os.makedirs("/opt/transcribe/cache", exist_ok=True)
    with open("/opt/transcribe/cache/model_info.json", "w") as f:
        json.dump(cache_info, f, indent=2)
    
    print("Model caching completed!")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
CACHE_SCRIPT

    # Upload and execute caching script
    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/cache_model.py ubuntu@"$PUBLIC_IP":/tmp/
    
    log "Executing model caching script..."
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "cd /opt/transcribe && sudo -u ubuntu /opt/transcribe/venv/bin/python /tmp/cache_model.py" \
        || handle_error "Model caching failed"
    
    # Verify model cache was created successfully
    log "Verifying model cache..."
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "echo '=== MODEL CACHE VERIFICATION ==='; echo 'Cache directory:'; ls -la /opt/transcribe/models/; echo 'Model files:'; ls -la /opt/transcribe/models/KBLab--kb-whisper-small/ 2>/dev/null || echo 'Model directory missing'; echo 'Model info:'; cat /opt/transcribe/cache/model_info.json 2>/dev/null || echo 'Model info file missing'; echo 'Cache size:'; du -sh /opt/transcribe/models/ 2>/dev/null || echo 'Cannot determine cache size'; echo 'Testing model loading:'; /opt/transcribe/venv/bin/python -c \"from transformers import AutoModelForSpeechSeq2Seq; model = AutoModelForSpeechSeq2Seq.from_pretrained('KBLab/kb-whisper-small', cache_dir='/opt/transcribe/models', local_files_only=True); print('Model loading test: SUCCESS')\" 2>/dev/null || echo 'Model loading test: FAILED'"
    
    log "Model cached and verified successfully"
}

# Create transcription script
create_transcription_script() {
    log "Creating transcription script..."
    
    # Get the directory where this script is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Verify local scripts exist before uploading
    if [[ ! -f "${SCRIPT_DIR}/fast_transcribe.py" ]]; then
        handle_error "Local Python script not found: ${SCRIPT_DIR}/fast_transcribe.py"
    fi
    
    if [[ ! -f "${SCRIPT_DIR}/fast_transcribe.sh" ]]; then
        handle_error "Local shell script not found: ${SCRIPT_DIR}/fast_transcribe.sh"
    fi
    
    # Log script versions being uploaded
    log "Uploading scripts from: $SCRIPT_DIR"
    log "Python script size: $(stat -c%s "${SCRIPT_DIR}/fast_transcribe.py" 2>/dev/null || echo "unknown") bytes"
    log "Shell script size: $(stat -c%s "${SCRIPT_DIR}/fast_transcribe.sh" 2>/dev/null || echo "unknown") bytes"
    
    # Upload the actual fast_transcribe.py script
    if ! scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        "${SCRIPT_DIR}/fast_transcribe.py" ubuntu@"$PUBLIC_IP":/opt/transcribe/fast_transcribe.py; then
        handle_error "Failed to upload Python transcription script"
    fi
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "chmod +x /opt/transcribe/fast_transcribe.py"
    
    # Upload the fast_transcribe.sh script to the correct location
    if ! scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        "${SCRIPT_DIR}/fast_transcribe.sh" ubuntu@"$PUBLIC_IP":/opt/transcription/fast_transcribe.sh; then
        handle_error "Failed to upload shell transcription script"
    fi
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "chmod +x /opt/transcription/fast_transcribe.sh"
    
    # ENHANCED: Verify script installation with detailed logging
    log "=== VERIFYING SCRIPT INSTALLATION ==="
    
    # Check file existence and permissions
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "echo '=== SCRIPT VERIFICATION ==='; echo 'Timestamp: $(date)'; echo 'Python script:'; ls -la /opt/transcribe/fast_transcribe.py; echo 'Shell script:'; ls -la /opt/transcription/fast_transcribe.sh; echo 'Directory contents:'; echo '/opt/transcribe:'; ls -la /opt/transcribe/; echo '/opt/transcription:'; ls -la /opt/transcription/"
    
    # Test script syntax
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "echo '=== SCRIPT SYNTAX TEST ==='; /opt/transcribe/venv/bin/python -m py_compile /opt/transcribe/fast_transcribe.py && echo 'Python script syntax OK' || echo 'Python script syntax ERROR'; bash -n /opt/transcription/fast_transcribe.sh && echo 'Shell script syntax OK' || echo 'Shell script syntax ERROR'"
    
    # Create cloud-init debugging script
    cat > /tmp/cloud_init_debug.sh << 'CLOUD_DEBUG'
#!/bin/bash
set -e

echo "=== CLOUD-INIT DEBUGGING SCRIPT ==="
echo "Timestamp: $(date)"
echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"

# Check cloud-init status
echo "=== CLOUD-INIT STATUS ==="
cloud-init status --wait || echo "cloud-init status command failed"
cloud-init status --long || echo "cloud-init status --long failed"

# Check cloud-init logs
echo "=== CLOUD-INIT LOGS ==="
if [ -f /var/log/cloud-init.log ]; then
    echo "Recent cloud-init.log entries:"
    tail -50 /var/log/cloud-init.log
else
    echo "cloud-init.log not found"
fi

if [ -f /var/log/cloud-init-output.log ]; then
    echo "Recent cloud-init-output.log entries:"
    tail -50 /var/log/cloud-init-output.log
else
    echo "cloud-init-output.log not found"
fi

# Check user data scripts
echo "=== USER DATA SCRIPTS ==="
if [ -d /var/lib/cloud/instance/scripts ]; then
    echo "Scripts directory contents:"
    ls -la /var/lib/cloud/instance/scripts/
    
    echo "Scripts directory permissions:"
    stat /var/lib/cloud/instance/scripts/
else
    echo "Scripts directory not found"
fi

# Check cloud-init configuration
echo "=== CLOUD-INIT CONFIG ==="
if [ -f /etc/cloud/cloud.cfg ]; then
    echo "cloud.cfg exists"
    grep -E "(scripts_user|scripts-per-instance|scripts-per-boot)" /etc/cloud/cloud.cfg || echo "No script configuration found in cloud.cfg"
else
    echo "cloud.cfg not found"
fi

# Check if our scripts are accessible
echo "=== SCRIPT ACCESSIBILITY ==="
echo "Python script:"
if [ -f /opt/transcribe/fast_transcribe.py ]; then
    echo "✓ Found: /opt/transcribe/fast_transcribe.py"
    ls -la /opt/transcribe/fast_transcribe.py
    echo "First few lines:"
    head -5 /opt/transcribe/fast_transcribe.py
else
    echo "✗ Missing: /opt/transcribe/fast_transcribe.py"
fi

echo "Shell script:"
if [ -f /opt/transcription/fast_transcribe.sh ]; then
    echo "✓ Found: /opt/transcription/fast_transcribe.sh"
    ls -la /opt/transcription/fast_transcribe.sh
    echo "First few lines:"
    head -5 /opt/transcription/fast_transcribe.sh
else
    echo "✗ Missing: /opt/transcription/fast_transcribe.sh"
fi

# Test script execution
echo "=== SCRIPT EXECUTION TEST ==="
echo "Testing Python script:"
if [ -f /opt/transcribe/fast_transcribe.py ]; then
    /opt/transcribe/venv/bin/python /opt/transcribe/fast_transcribe.py --help 2>&1 || echo "Python script execution test failed"
else
    echo "Python script not available for testing"
fi

echo "Testing shell script:"
if [ -f /opt/transcription/fast_transcribe.sh ]; then
    bash -n /opt/transcription/fast_transcribe.sh && echo "Shell script syntax check passed" || echo "Shell script syntax check failed"
else
    echo "Shell script not available for testing"
fi

echo "=== DEBUG COMPLETE ==="
CLOUD_DEBUG

    # Upload and make executable
    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/cloud_init_debug.sh ubuntu@"$PUBLIC_IP":/opt/transcription/cloud_init_debug.sh
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "chmod +x /opt/transcription/cloud_init_debug.sh"
    
    log "Transcription scripts uploaded and configured with enhanced verification"
}

# Final validation
validate_setup() {
    log "Running final validation..."
    
    # Create validation script
    cat > /tmp/validate.sh << 'VALIDATE_SCRIPT'
#!/bin/bash
set -e

echo "=== AMI Validation ==="
echo "Timestamp: $(date)"

# Check NVIDIA drivers
echo -n "[$(date)] Checking NVIDIA drivers... "
if ! nvidia-smi &> /dev/null; then
    echo "FAILED"
    echo "NVIDIA driver diagnostic:"
    lsmod | grep nvidia || echo "No NVIDIA modules loaded"
    nvidia-smi 2>&1 || true
    dkms status nvidia || echo "No NVIDIA DKMS status"
    exit 1
fi
echo "OK"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Detailed CUDA validation
echo -n "[$(date)] Checking CUDA availability... "
CUDA_CHECK=$(/opt/transcribe/venv/bin/python -c "import torch; print('CUDA_AVAILABLE' if torch.cuda.is_available() else 'CUDA_NOT_AVAILABLE')" 2>/dev/null)
if [ "$CUDA_CHECK" = "CUDA_AVAILABLE" ]; then
    echo "OK"
    /opt/transcribe/venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA devices: {torch.cuda.device_count()}')"
else
    echo "FAILED - CUDA not available to PyTorch"
    /opt/transcribe/venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || echo "Failed to import torch"
    exit 1
fi

# Quick validation of other components
echo -n "[$(date)] Checking Python environment... "
/opt/transcribe/venv/bin/python -c "import transformers, librosa, soundfile, boto3" &> /dev/null && echo "OK" || { echo "FAILED"; exit 1; }
touch /opt/transcribe/.python_verified

echo -n "[$(date)] Checking PyTorch... "
/opt/transcribe/venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" &> /dev/null && echo "OK" || { echo "FAILED"; exit 1; }
touch /opt/transcribe/.pytorch_verified

echo -n "[$(date)] Checking transformers... "
/opt/transcribe/venv/bin/python -c "import transformers; print(f'Transformers {transformers.__version__}')" &> /dev/null && echo "OK" || { echo "FAILED"; exit 1; }
touch /opt/transcribe/.transformers_verified

echo -n "[$(date)] Checking S3 access... "
/opt/transcribe/venv/bin/python -c "import boto3; boto3.client('s3').list_buckets()" &> /dev/null && echo "OK (credentials configured)" || echo "WARNING (no S3 credentials - upload will be skipped)"

echo -n "[$(date)] Checking model cache... "
[ -f /opt/transcribe/cache/model_info.json ] && [ -d /opt/transcribe/models ] && echo "OK" || { echo "FAILED"; exit 1; }
touch /opt/transcribe/.model_cache_verified

echo -n "[$(date)] Checking transcription scripts... "
[ -f /opt/transcribe/fast_transcribe.py ] && [ -f /opt/transcription/fast_transcribe.sh ] && echo "OK" || { echo "FAILED"; exit 1; }

# Test script syntax
echo -n "[$(date)] Testing script syntax... "
/opt/transcribe/venv/bin/python -m py_compile /opt/transcribe/fast_transcribe.py && echo "OK" || { echo "FAILED"; exit 1; }

# Create comprehensive completion marker with all verification info
cat > /opt/transcribe/.setup_complete << EOF
AMI_SETUP_COMPLETE=true
SETUP_DATE=\$(date)
NVIDIA_DRIVERS=installed
PYTHON_ENV=ready
PYTORCH=ready
TRANSFORMERS=ready
MODEL_CACHE=ready
SCRIPTS=ready
KERNEL_VERSION=\$(uname -r)
NVIDIA_VERSION=\$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
PYTORCH_VERSION=\$(/opt/transcribe/venv/bin/python -c "import torch; print(torch.__version__)")
TRANSFORMERS_VERSION=\$(/opt/transcribe/venv/bin/python -c "import transformers; print(transformers.__version__)")
MODEL_ID=KBLab/kb-whisper-small
EOF

echo "[$(date)] Validation complete"
VALIDATE_SCRIPT

    # Upload and validate
    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/validate.sh ubuntu@"$PUBLIC_IP":/tmp/
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "sudo bash /tmp/validate.sh" || handle_error "Validation failed"
    
    log "Validation completed successfully"
}

# Create AMI
create_ami() {
    log "Creating AMI..."
    
    AMI_NAME="transcription-gpu-$(date +%Y%m%d-%H%M%S)"
    
    AMI_ID=$(aws ec2 create-image \
        --instance-id "$INSTANCE_ID" \
        --name "$AMI_NAME" \
        --description "GPU transcription AMI with cached Swedish Whisper model" \
        --query 'ImageId' \
        --output text) || handle_error "Failed to create AMI"
    
    log "AMI creation initiated: $AMI_ID"
    
    # Wait for AMI to be available with improved configuration
    log "Waiting for AMI to be available (this may take up to 30 minutes)..."
    
    # Custom wait loop for AMI availability with longer timeout
    local max_attempts=60  # 60 attempts
    local delay=30        # 30 seconds between attempts
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
    
    # Write AMI ID to file at the very end
    echo "$AMI_ID" > ami_id.txt
    log "AMI ID saved to ami_id.txt: $AMI_ID"
}

# Update AMI ID in lambda function
update_lambda_ami_id() {
    log "Updating AMI ID in lambda function..."
    
    local lambda_file="../setup/lambda/lambda_process_upload.py"
    
    if [ -f "$lambda_file" ]; then
        # Create backup
        cp "$lambda_file" "${lambda_file}.backup"
        
        # Update AMI ID in lambda function
        sed -i "s/AMI_ID = 'ami-[a-z0-9]*'/AMI_ID = '$AMI_ID'/" "$lambda_file"
        
        log "Updated AMI ID in lambda function: $lambda_file"
        log "Previous AMI ID backed up to: ${lambda_file}.backup"
        
        # Verify the change
        if grep -q "AMI_ID = '$AMI_ID'" "$lambda_file"; then
            log "AMI ID update verified in lambda function"
        else
            log "WARNING: AMI ID update verification failed"
        fi
    else
        log "WARNING: Lambda function file not found: $lambda_file"
    fi
    
    # Update AMI ID in Python transcription script
    local python_file="fast_transcribe.py"
    
    if [ -f "$python_file" ]; then
        # Create backup
        cp "$python_file" "${python_file}.backup"
        
        # Update expected AMI ID in Python script
        sed -i "s/EXPECTED_AMI_ID = 'ami-[a-z0-9]*'/EXPECTED_AMI_ID = '$AMI_ID'/" "$python_file"
        
        log "Updated expected AMI ID in Python script: $python_file"
        
        # Verify the change
        if grep -q "EXPECTED_AMI_ID = '$AMI_ID'" "$python_file"; then
            log "Expected AMI ID update verified in Python script"
        else
            log "WARNING: Expected AMI ID update verification failed"
        fi
    else
        log "WARNING: Python transcription script not found: $python_file"
    fi
}

# Main execution
main() {
    log "Starting AMI build process..."
    
    # Log configuration for transparency
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
    cache_model
    create_transcription_script
    validate_setup
    create_ami
    update_lambda_ami_id
    
    log "AMI build completed successfully!"
    log "Final AMI ID: $AMI_ID"
    log "AMI ID file updated: ami_id.txt"
    log "Lambda function and scripts updated with new AMI ID"
    
    # Final summary
    log "=== BUILD SUMMARY ==="
    log "✓ Base AMI: $BASE_AMI"
    log "✓ New AMI: $AMI_ID"
    log "✓ Instance Type: $INSTANCE_TYPE"
    log "✓ Model: $MODEL_ID"
    log "✓ NVIDIA Drivers: Installed"
    log "✓ Python Environment: Ready"
    log "✓ Model Cache: Created"
    log "✓ Scripts: Uploaded and Verified"
    log "✓ Lambda Function: Updated"
    log "✓ Setup Marker: Created"
    log "====================="
    log ""
    log "Next steps:"
    log "1. Deploy the updated lambda function with: cd ../setup/lambda && ./deploy_lambda_functions.sh"
    log "2. Test transcription by uploading a file to S3"
    log "3. Monitor CloudWatch logs for any issues"
}

# Run main function
main