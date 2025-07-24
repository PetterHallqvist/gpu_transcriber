#!/bin/bash

# Fast GPU Transcription - Simple & Elegant
# ==========================================
# Streamlined script for fast transcription using cached AMI

set -euo pipefail

# Configuration
AUDIO_FILE="${1:-}"
REGION="eu-north-1"
KEY_NAME="transcription-ec2"
SECURITY_GROUP="transcription-g4dn-sg"

# Logging function with timestamps
log_msg() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log_msg "=== Fast GPU Transcription ==="

# Validate inputs
if [[ -z "$AUDIO_FILE" ]]; then
    log_msg "Usage: $0 <audio_file>"
    exit 1
fi

if [[ ! -f "$AUDIO_FILE" ]]; then
    log_msg "Error: Audio file not found: $AUDIO_FILE"
    exit 1
fi

if [[ ! -f "ami_id.txt" ]]; then
    log_msg "Error: ami_id.txt not found. Run ./build_ami.sh first"
    exit 1
fi

AMI_ID=$(cat ami_id.txt)
log_msg "Using AMI: $AMI_ID"
log_msg "Audio file: $AUDIO_FILE"
log_msg "Region: $REGION"

# Validate key file exists
if [[ ! -f "${KEY_NAME}.pem" ]]; then
    log_msg "Error: SSH key file not found: ${KEY_NAME}.pem"
    exit 1
fi

# Global variables
INSTANCE_ID=""
PUBLIC_IP=""

# Cleanup function
cleanup() {
    if [[ -n "$INSTANCE_ID" ]]; then
        log_msg "Terminating instance: $INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null 2>&1 || true
    fi
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Launch instance
log_msg "Launching GPU instance (g4dn.xlarge)..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "g4dn.xlarge" \
    --key-name "$KEY_NAME" \
    --security-groups "$SECURITY_GROUP" \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' \
    --output text) || {
    log_msg "Error: Failed to launch instance"
    exit 1
}

log_msg "Instance launched: $INSTANCE_ID"

# Wait for instance to be running
log_msg "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text --region "$REGION")

log_msg "Instance running at: $PUBLIC_IP"

# Wait for SSH to be ready
log_msg "Waiting for SSH connection..."
for i in {1..30}; do
    if ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
           ubuntu@"$PUBLIC_IP" "echo 'SSH ready'" >/dev/null 2>&1; then
        log_msg "SSH connection established"
        break
    fi
    if [[ $i -eq 30 ]]; then
        log_msg "Error: SSH timeout after 150 seconds"
        exit 1
    fi
    sleep 5
done

# Upload files
log_msg "Uploading audio file..."
scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
    "$AUDIO_FILE" ubuntu@"$PUBLIC_IP":/home/ubuntu/ || {
    log_msg "Error: Failed to upload audio file"
    exit 1
}

log_msg "Uploading transcription script..."
scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
    fast_transcribe.py ubuntu@"$PUBLIC_IP":/home/ubuntu/ || {
    log_msg "Error: Failed to upload transcription script"
    exit 1
}

# Run transcription with comprehensive checks
log_msg "Running transcription on GPU instance..."
START_TIME=$(date +%s)

ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" << 'EOF'
set -e

# Logging function
log_msg() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log_msg "=== Remote Instance Setup Verification ==="

# Check if AMI setup is complete
if [[ ! -f "/opt/transcribe/.setup_complete" ]]; then
    log_msg "Error: AMI setup incomplete - .setup_complete marker not found"
    exit 1
fi

log_msg "AMI setup verified - .setup_complete marker found"

# Verify NVIDIA drivers
log_msg "Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    log_msg "Error: nvidia-smi not found"
    exit 1
fi

# Get GPU information
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
log_msg "GPU Information:"
log_msg "  $GPU_INFO"

# Check GPU memory
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
log_msg "Available GPU memory: ${GPU_MEMORY}MB"

if [[ $GPU_MEMORY -lt 1000 ]]; then
    log_msg "Warning: Low GPU memory available (${GPU_MEMORY}MB)"
fi

# Verify CUDA installation
log_msg "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    log_msg "CUDA version: $CUDA_VERSION"
else
    log_msg "Warning: nvcc not found in PATH"
fi

# Navigate to transcribe directory
cd /opt/transcribe

# Verify virtual environment exists
if [[ ! -d "venv" ]]; then
    log_msg "Error: Virtual environment not found at /opt/transcribe/venv"
    exit 1
fi

log_msg "Virtual environment found"

# Activate virtual environment
log_msg "Activating virtual environment..."
source venv/bin/activate

# Verify Python and dependencies
log_msg "Checking Python environment..."
PYTHON_VERSION=$(python3 --version)
log_msg "Python version: $PYTHON_VERSION"

# Check PyTorch installation and CUDA support
log_msg "Checking PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA not available in PyTorch')
" || {
    log_msg "Error: PyTorch verification failed"
    exit 1
}

# Verify transformers library
log_msg "Checking transformers library..."
python3 -c "
import transformers
print(f'Transformers version: {transformers.__version__}')
" || {
    log_msg "Error: Transformers library not found"
    exit 1
}

# Verify model cache directory
if [[ ! -d "/opt/transcribe/models" ]]; then
    log_msg "Error: Model cache directory not found"
    exit 1
fi

MODEL_COUNT=$(find /opt/transcribe/models -type f | wc -l)
log_msg "Model cache directory verified ($MODEL_COUNT files)"

# Check available disk space
DISK_SPACE=$(df -h /opt/transcribe | tail -1 | awk '{print $4}')
log_msg "Available disk space: $DISK_SPACE"

# Copy and verify transcription script
cp /home/ubuntu/fast_transcribe.py ./
if [[ ! -f "fast_transcribe.py" ]]; then
    log_msg "Error: Failed to copy transcription script"
    exit 1
fi

log_msg "Transcription script copied successfully"

# Run transcription
log_msg "=== Starting Transcription ==="
AUDIO_BASENAME=$(basename "/home/ubuntu/$(ls /home/ubuntu/*.wav /home/ubuntu/*.mp3 /home/ubuntu/*.m4a /home/ubuntu/*.flac 2>/dev/null | head -1)" 2>/dev/null || echo "unknown")
log_msg "Audio file: $AUDIO_BASENAME"

python3 fast_transcribe.py "/home/ubuntu/$AUDIO_BASENAME" || {
    log_msg "Error: Transcription failed"
    exit 1
}

log_msg "Transcription completed successfully!"

# Show GPU memory usage after transcription
log_msg "Final GPU memory status:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | head -1 | while read used free; do
    log_msg "  Used: ${used}MB, Free: ${free}MB"
done

EOF

REMOTE_EXIT_CODE=$?
if [[ $REMOTE_EXIT_CODE -ne 0 ]]; then
    log_msg "Error: Remote transcription failed with exit code $REMOTE_EXIT_CODE"
    exit 1
fi

# Download results
log_msg "Downloading results..."
scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
    ubuntu@"$PUBLIC_IP":/opt/transcribe/transcription_*.txt . 2>/dev/null || {
    log_msg "Warning: No result files found to download"
    # Try to get any text files from transcribe directory
    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP":/opt/transcribe/*.txt . 2>/dev/null || true
}

# Upload results to S3 (if available)
RESULT_FILES=$(ls transcription_*.txt 2>/dev/null || true)
if [[ -n "$RESULT_FILES" ]]; then
    log_msg "Uploading results to S3..."
    for file in $RESULT_FILES; do
        TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
        AUDIO_BASENAME=$(basename "$AUDIO_FILE" | sed 's/\.[^.]*$//')
        S3_KEY="results/${TIMESTAMP}/transcription_${AUDIO_BASENAME}.txt"
        
        if aws s3 cp "$file" "s3://transcription-curevo/${S3_KEY}" \
           --metadata "audio_file=${AUDIO_BASENAME},timestamp=${TIMESTAMP}" \
           --region "$REGION" 2>/dev/null; then
            log_msg "Uploaded to S3: s3://transcription-curevo/${S3_KEY}"
        else
            log_msg "Warning: Failed to upload $file to S3 (continuing without S3)"
        fi
    done
else
    log_msg "No result files to upload to S3"
fi

# Show results
TOTAL_TIME=$(($(date +%s) - START_TIME))
log_msg ""
log_msg "=== Transcription Complete ==="
log_msg "Total time: ${TOTAL_TIME} seconds"
log_msg "Instance: $INSTANCE_ID"

# Show result files
RESULT_FILES=$(ls transcription_*.txt 2>/dev/null || true)
if [[ -n "$RESULT_FILES" ]]; then
    log_msg "Result files:"
    for file in $RESULT_FILES; do
        log_msg "  - $file"
    done
    
    # Show latest transcription
    LATEST_FILE=$(ls -t transcription_*.txt 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_FILE" ]]; then
        log_msg ""
        log_msg "=== Latest Transcription ==="
        log_msg "File: $LATEST_FILE"
        echo ""
        # Show just the transcription part
        sed -n '/=== TRANSCRIPTION ===/,$p' "$LATEST_FILE" | tail -n +2
        
        # Show summary from file
        log_msg ""
        log_msg "=== Performance Summary ==="
        grep "Device:" "$LATEST_FILE" || true
        grep "GPU:" "$LATEST_FILE" || true
        grep "Load time:" "$LATEST_FILE" || true
        grep "Transcription time:" "$LATEST_FILE" || true
        grep "Total time:" "$LATEST_FILE" || true
        grep "Text length:" "$LATEST_FILE" || true
        
        # Show S3 upload status if available
        if aws s3 ls s3://transcription-curevo/results/ >/dev/null 2>&1; then
            log_msg "S3 upload: Available ✓"
        else
            log_msg "S3 upload: Not configured (local results only)"
        fi
    fi
else
    log_msg "No result files downloaded"
fi

log_msg ""
log_msg "Fast transcription completed!" 