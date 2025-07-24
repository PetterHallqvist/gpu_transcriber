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

# Install essential packages (without GPU drivers)
retry_apt_operation "apt-get install -y \
    build-essential \
    linux-headers-generic \
    dkms \
    python3-pip \
    python3-venv \
    python3-dev \
    ffmpeg \
    git \
    curl \
    wget" || exit 1

# Verify kernel headers
if [ ! -d "/lib/modules/$KERNEL_VERSION/build" ]; then
    echo "ERROR: Kernel headers directory not found at /lib/modules/$KERNEL_VERSION/build"
    exit 1
fi

# Install NVIDIA drivers with DKMS support
echo "[$(date)] Installing NVIDIA drivers..."
retry_apt_operation "DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-drivers-common" || exit 1
echo "[$(date)] Available NVIDIA drivers:"
ubuntu-drivers devices

# Install specific NVIDIA driver version for better compatibility
retry_apt_operation "DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-535 nvidia-dkms-535" || exit 1

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
        
        # Create NVIDIA device nodes
        if [ ! -e /dev/nvidia0 ]; then
            NVIDIA_MAJOR=$(grep nvidia /proc/devices | awk '{print $1}')
            if [ -n "$NVIDIA_MAJOR" ]; then
                mknod -m 666 /dev/nvidia0 c $NVIDIA_MAJOR 0
                mknod -m 666 /dev/nvidiactl c $NVIDIA_MAJOR 255
                echo "[$(date)] NVIDIA device nodes created"
            fi
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
chown -R ubuntu:ubuntu /opt/transcribe

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

model_id = "KBLab/kb-whisper-small"
cache_dir = "/opt/transcribe/models"

print(f"Downloading and caching model: {model_id}")

try:
    # Download model (CPU mode for caching)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    print("Model downloaded successfully")
    
    # Download processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir
    )
    print("Processor downloaded successfully")
    
    # Test loading from cache
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
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "cd /opt/transcribe && sudo -u ubuntu /opt/transcribe/venv/bin/python /tmp/cache_model.py" \
        || handle_error "Model caching failed"
    
    log "Model cached successfully"
}

# Create transcription script
create_transcription_script() {
    log "Creating transcription script..."
    
    cat > /tmp/transcribe.py << 'TRANSCRIBE_SCRIPT'
#!/opt/transcribe/venv/bin/python3
import os
import sys
import time
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

def main():
    if len(sys.argv) != 2:
        print("Usage: transcribe.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Configuration
    model_id = "KBLab/kb-whisper-small"
    cache_dir = "/opt/transcribe/models"
    
    print("Loading model from cache...")
    start_time = time.time()
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Using device: {device}")
    
    # Load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        torch_dtype=torch_dtype
    )
    
    if device == "cuda":
        model = model.to(device)
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True
    )
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=8,
        torch_dtype=torch_dtype,
        device=device
    )
    
    # Transcribe
    print(f"Transcribing {audio_file}...")
    start_time = time.time()
    
    result = pipe(audio_file, return_timestamps=True)
    transcription = result["text"]
    
    duration = time.time() - start_time
    print(f"Transcription completed in {duration:.2f} seconds")
    
    # Save result
    output_file = os.path.splitext(audio_file)[0] + "_transcription.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcription)
    
    print(f"Transcription saved to {output_file}")
    print("\n--- Transcription ---")
    print(transcription)

if __name__ == "__main__":
    main()
TRANSCRIBE_SCRIPT

    # Upload script
    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/transcribe.py ubuntu@"$PUBLIC_IP":/opt/transcribe/scripts/
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "chmod +x /opt/transcribe/scripts/transcribe.py"
    
    log "Transcription script created"
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

echo -n "[$(date)] Checking S3 access... "
/opt/transcribe/venv/bin/python -c "import boto3; boto3.client('s3').list_buckets()" &> /dev/null && echo "OK (credentials configured)" || echo "WARNING (no S3 credentials - upload will be skipped)"

echo -n "[$(date)] Checking model cache... "
[ -f /opt/transcribe/cache/model_info.json ] && [ -d /opt/transcribe/models ] && echo "OK" || { echo "FAILED"; exit 1; }

# Create completion marker
touch /opt/transcribe/.setup_complete

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
    echo "$AMI_ID" > ami_id.txt
    
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
}

# Main execution
main() {
    log "Starting AMI build process..."
    
    validate_prerequisites
    launch_instance
    establish_ssh
    setup_instance
    cache_model
    create_transcription_script
    validate_setup
    create_ami
    
    log "AMI build completed successfully!"
    log "AMI ID: $AMI_ID"
    log "Saved to: ami_id.txt"
}

# Run main function
main