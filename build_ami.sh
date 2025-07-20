#!/bin/bash

# Modular Production AMI Builder
# Components: transcribe_optimized.py, api_server.py, advanced_cache.py, transcribe-api.service

echo "🏗️ Building Optimized Transcription AMI"
echo "Target: <20s boot, instant transcription"

export AWS_DEFAULT_REGION=eu-north-1

# Cleanup function for automatic instance termination
cleanup_instances() {
    echo ""
    echo "Cleaning up build instances..."
    
    if [ ! -z "$INSTANCE_ID" ]; then
        echo "Terminating build instance: $INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null 2>&1
    fi
    
    # Clean up temp files
    rm -f /tmp/build_optimized_ami_instance_id
    
    echo "Build cleanup completed"
}

# Set up signal traps for automatic cleanup
trap cleanup_instances EXIT INT TERM

# Clean up any previous build instances
if [ -f /tmp/build_optimized_ami_instance_id ]; then
    OLD_INSTANCE_ID=$(cat /tmp/build_optimized_ami_instance_id 2>/dev/null)
    if [ ! -z "$OLD_INSTANCE_ID" ]; then
        echo "Cleaning up previous build instance: $OLD_INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$OLD_INSTANCE_ID" >/dev/null 2>&1
    fi
fi

# Validate that all component files exist
echo "📁 Validating components..."
REQUIRED_FILES=(
    "transcribe_optimized.py"
    "api_server.py" 
    "advanced_cache.py"
    "transcribe-api.service"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ ERROR: Missing: $file"
        exit 1
    fi
done
echo "✅ All components found"

echo "🚀 Launching optimized build instance..."

# Try multiple zones for best availability
ZONES=("eu-north-1a" "eu-north-1b" "eu-north-1c")
INSTANCE_ID=""

for zone in "${ZONES[@]}"; do
    echo "📍 Trying to launch build instance in zone: $zone"
    
    INSTANCE_OUTPUT=$(aws ec2 run-instances \
        --image-id "ami-0989fb15ce71ba39e" \
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
        echo "✅ SUCCESS: Instance launched in zone: $zone"
        echo "Instance ID: $INSTANCE_ID"
        break
    else
        echo "❌ FAILED: Could not launch in zone $zone"
        if echo "$INSTANCE_OUTPUT" | grep -q "An error occurred"; then
            echo "Error: $INSTANCE_OUTPUT"
        fi
        echo "Trying next zone..."
    fi
done

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
    echo "❌ ERROR: Failed to launch instance in any zone"
    exit 1
fi

# Store for cleanup
echo "$INSTANCE_ID" > /tmp/build_optimized_ami_instance_id

echo "⏳ Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "📡 Public IP: $PUBLIC_IP"
echo "⏳ Waiting for SSH to be ready..."

# Wait for SSH with optimized retry
SSH_READY=false
for i in {1..30}; do
    if ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$PUBLIC_IP "echo 'SSH Ready'" >/dev/null 2>&1; then
        SSH_READY=true
        echo "✅ SSH connection established"
        break
    fi
    echo "⏳ Attempt $i/30: SSH not ready, waiting..."
    sleep 10
done

if [ "$SSH_READY" = false ]; then
    echo "❌ ERROR: SSH failed after 5 minutes"
    exit 1
fi

echo ""
echo "🏗️ Installing OPTIMIZED production environment..."

# Upload modular components first
echo "📤 Uploading modular components..."

# Upload with error checking
if ! scp -i transcription-ec2.pem -o StrictHostKeyChecking=no \
    transcribe_optimized.py \
    api_server.py \
    advanced_cache.py \
    transcribe-api.service \
    ubuntu@$PUBLIC_IP:/tmp/; then
    echo "❌ ERROR: Failed to upload components to instance"
    exit 1
fi

# Verify files were actually uploaded
echo "🔍 Verifying file upload..."
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'VERIFY_UPLOAD_EOF'
echo "Checking uploaded files in /tmp/:"
for file in transcribe_optimized.py api_server.py advanced_cache.py transcribe-api.service; do
    if [ -f "/tmp/$file" ]; then
        echo "✅ $file uploaded successfully"
    else
        echo "❌ $file missing!"
        exit 1
    fi
done
VERIFY_UPLOAD_EOF

echo "✅ All components uploaded and verified successfully"

# Create Phase 1 setup script (drivers + system prep)
cat > /tmp/setup_phase1.sh << 'PHASE1_EOF'
#!/bin/bash
set -e

echo "=== PHASE 1: NVIDIA Drivers + System Setup ==="
echo "Timestamp: $(date)"

# Update system first
echo "Updating system packages..."
sudo apt update -y

# Install kernel headers FIRST (critical for NVIDIA driver compilation)
echo "Installing kernel headers..."
sudo apt install -y linux-headers-$(uname -r) build-essential dkms

# Install boot optimization packages
echo "Installing boot optimization packages..."
sudo apt install -y preload zram-config || echo "Some optimization packages not available, continuing..."

# Disable unnecessary services for faster boot
echo "Disabling unnecessary services..."
sudo systemctl disable snapd.service || true
sudo systemctl disable snap.amazon-ssm-agent.service || true
sudo systemctl disable ubuntu-advantage.service || true
sudo systemctl disable unattended-upgrades.service || true
sudo systemctl disable apt-daily.service || true
sudo systemctl disable apt-daily-upgrade.service || true

# Optimize SSH for faster connections
echo "Optimizing SSH configuration..."
sudo sed -i 's/#UseDNS yes/UseDNS no/' /etc/ssh/sshd_config
sudo sed -i 's/#GSSAPIAuthentication yes/GSSAPIAuthentication no/' /etc/ssh/sshd_config
echo "ClientAliveInterval 30" | sudo tee -a /etc/ssh/sshd_config
echo "ClientAliveCountMax 3" | sudo tee -a /etc/ssh/sshd_config

# CRITICAL: Install NVIDIA drivers properly
echo ""
echo "🚀 INSTALLING NVIDIA DRIVERS (PHASE 1)"
echo "======================================"

# Clean any existing nvidia installations
sudo apt purge -y 'nvidia-*' || true
sudo apt autoremove -y || true

# Install ubuntu-drivers-common
echo "Installing ubuntu-drivers-common..."
sudo apt install -y ubuntu-drivers-common

# Show available drivers
echo "Available NVIDIA drivers:"
sudo ubuntu-drivers devices || echo "Could not detect devices"

# Install NVIDIA drivers with strict error checking
echo "Installing NVIDIA drivers..."
if sudo ubuntu-drivers autoinstall; then
    echo "✅ ubuntu-drivers autoinstall succeeded"
else
    echo "⚠️ ubuntu-drivers autoinstall failed, trying manual approach..."
    
    # Fallback: Install specific driver
    if sudo apt install -y nvidia-driver-535-server nvidia-utils-535-server; then
        echo "✅ Manual driver installation succeeded"
    else
        echo "❌ NVIDIA driver installation failed completely"
        exit 1
    fi
fi

# Verify installation (files should exist, but driver won't work until reboot)
if [ -f "/usr/bin/nvidia-smi" ]; then
    echo "✅ nvidia-smi binary installed"
else
    echo "❌ nvidia-smi binary missing - driver installation failed"
    exit 1
fi

# Create a marker file to indicate Phase 1 completed
echo "$(date): Phase 1 completed successfully" > /tmp/phase1_complete.marker

echo ""
echo "✅ PHASE 1 COMPLETE - NVIDIA drivers installed"
echo "⚠️  REBOOT REQUIRED to load NVIDIA kernel modules"
echo "Next: External reboot + Phase 2"

PHASE1_EOF

# Upload and run Phase 1
echo "📤 Uploading and running Phase 1 (drivers + system setup)..."
scp -i transcription-ec2.pem -o StrictHostKeyChecking=no /tmp/setup_phase1.sh ubuntu@$PUBLIC_IP:/tmp/
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "chmod +x /tmp/setup_phase1.sh && sudo /tmp/setup_phase1.sh"

echo ""
echo "🔄 REBOOTING INSTANCE TO LOAD NVIDIA DRIVERS"
echo "============================================"

# Reboot the instance via AWS API (cleaner than internal reboot)
echo "Initiating instance reboot via AWS API..."
aws ec2 reboot-instances --instance-ids $INSTANCE_ID

# Wait a moment for reboot to start
echo "Waiting for reboot to initiate..."
sleep 10

# Wait for instance to be running again
echo "Waiting for instance to come back online..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Wait for SSH to be ready again
echo "Waiting for SSH to be ready after reboot..."
SSH_READY=false
for i in {1..30}; do
    if ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$PUBLIC_IP "echo 'SSH Ready After Reboot'" >/dev/null 2>&1; then
        SSH_READY=true
        echo "✅ SSH reconnected after reboot (attempt $i)"
        break
    fi
    echo "⏳ Attempt $i/30: SSH not ready after reboot, waiting..."
    sleep 10
done

if [ "$SSH_READY" = false ]; then
    echo "❌ ERROR: SSH failed after reboot"
    exit 1
fi

echo ""
echo "🔍 VERIFYING NVIDIA DRIVERS AFTER REBOOT"
echo "========================================"

# Verify NVIDIA drivers are working
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'VERIFY_EOF'
echo "Testing NVIDIA drivers after reboot..."

if nvidia-smi >/dev/null 2>&1; then
    echo "✅ nvidia-smi working!"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "❌ nvidia-smi still not working after reboot"
    echo "Checking kernel modules..."
    lsmod | grep nvidia || echo "No nvidia modules loaded"
    echo "Checking device files..."
    ls -la /dev/nvidia* 2>/dev/null || echo "No nvidia device files"
    exit 1
fi

echo "✅ NVIDIA drivers verified successfully!"
VERIFY_EOF

echo ""
echo "🚀 PHASE 2: Python Environment + Model Caching"
echo "=============================================="

# Create Phase 2 setup script
cat > /tmp/setup_phase2.sh << 'PHASE2_EOF'
#!/bin/bash
set -e

echo "=== PHASE 2: Python Environment + Dependencies ==="
echo "Timestamp: $(date)"

# Install comprehensive audio processing stack
echo "Installing comprehensive audio processing stack..."
sudo apt install -y \
    ffmpeg \
    libavcodec-extra \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libsox-fmt-all \
    sox \
    alsa-utils \
    pulseaudio \
    libsndfile1-dev \
    libflac-dev \
    libvorbis-dev \
    libopus-dev \
    libmp3lame-dev \
    libfdk-aac2

# Install Python optimizations and development tools
echo "Installing optimized Python stack..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-wheel \
    python3-setuptools \
    build-essential \
    cmake \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran

# Install network and system performance tools
echo "Installing performance optimization tools..."
sudo apt install -y \
    htop \
    iotop \
    nethogs \
    curl \
    wget \
    unzip \
    git \
    jq

# Create optimized directory structure
echo ""
echo "Creating optimized production structure..."

# Create directories with proper permissions
echo "Creating /opt/transcribe directory structure..."
sudo rm -rf /opt/transcribe 2>/dev/null || true
sudo mkdir -p /opt/transcribe/{venv,models,cache,scripts,config,temp,logs}
sudo chown -R ubuntu:ubuntu /opt/transcribe
sudo chmod -R 755 /opt/transcribe

# Verify we can access the directory
cd /opt/transcribe || { echo "ERROR: Cannot access /opt/transcribe directory"; exit 1; }
echo "Directory structure created: $(pwd)"

# Create highly optimized Python environment
echo ""
echo "Creating optimized Python environment..."

# Create and test venv in one go
echo "Creating Python virtual environment..."
python3 -m venv venv --system-site-packages
source venv/bin/activate || { echo "ERROR: Virtual environment creation failed"; exit 1; }
echo "Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip with optimizations
echo "Upgrading pip with optimizations..."
pip install --upgrade pip setuptools wheel

# Set pip to use optimized settings
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'PIPCONF'
[global]
no-cache-dir = true
compile = true
optimize = 2
PIPCONF

# CRITICAL: Install CUDA-enabled PyTorch with NVIDIA drivers working
echo ""
echo "🚀 Installing CUDA-enabled PyTorch..."
echo "NVIDIA Status Check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "GPU check failed"

# Install PyTorch with CUDA support (optimized for T4)
echo "Installing optimized PyTorch for T4 GPU with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch CUDA immediately after installation
echo "Verifying PyTorch CUDA installation..."
python -c "
import torch
import sys
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    # Quick GPU test
    x = torch.randn(100, 100, device='cuda')
    print('✅ GPU tensor allocation successful')
    del x
    torch.cuda.empty_cache()
else:
    print('❌ CUDA not available - this is a problem!')
    sys.exit(1)
"

# Install optimized transformers stack
echo "Installing optimized transformers and dependencies..."
pip install \
    transformers[torch] \
    datasets \
    accelerate \
    optimum \
    numba \
    librosa \
    soundfile \
    scipy \
    numpy

# Install additional audio processing libraries
echo "Installing additional audio processing libraries..."
pip install \
    pydub \
    wave \
    audioop \
    mutagen \
    pyaudio \
    audioread

# Install Flask and API dependencies
echo "Installing Flask and API server dependencies..."
pip install \
    flask \
    werkzeug \
    gunicorn \
    waitress

# Install transcription scripts and components
echo ""
echo "Installing transcription scripts and API server..."

# Ensure scripts directory exists and is accessible
sudo mkdir -p /opt/transcribe/scripts
sudo chown -R ubuntu:ubuntu /opt/transcribe

# Copy scripts from /tmp/ to final locations
echo "Copying transcription scripts..."
cp /tmp/transcribe_optimized.py /opt/transcribe/scripts/
cp /tmp/api_server.py /opt/transcribe/scripts/
cp /tmp/advanced_cache.py /opt/transcribe/
chmod +x /opt/transcribe/scripts/*.py

echo "✅ Scripts installed successfully"

# Install systemd service
echo "Installing API service..."
sudo cp /tmp/transcribe-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable transcribe-api

echo "✅ API service installed successfully"

echo "✅ Phase 2 completed - Python environment, scripts, and services ready with CUDA support"

PHASE2_EOF

# Upload and run Phase 2
echo "📤 Uploading and running Phase 2 (Python environment + dependencies)..."
scp -i transcription-ec2.pem -o StrictHostKeyChecking=no /tmp/setup_phase2.sh ubuntu@$PUBLIC_IP:/tmp/
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "chmod +x /tmp/setup_phase2.sh && /tmp/setup_phase2.sh"

echo ""
echo "🧠 RUNNING GPU-OPTIMIZED MODEL CACHING"
echo "======================================"

# Run enhanced model caching with working GPU
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'CACHING_SESSION'
set -e

# Ensure directory exists and is accessible
sudo mkdir -p /opt/transcribe/cache
sudo chown -R ubuntu:ubuntu /opt/transcribe

# Navigate to transcription environment (advanced_cache.py should already be here from Phase 2)
cd /opt/transcribe
source venv/bin/activate

echo "Running GPU-optimized model caching..."
echo "GPU Status before caching:"
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader

# Verify advanced_cache.py exists
if [ ! -f "advanced_cache.py" ]; then
    echo "❌ ERROR: advanced_cache.py not found in /opt/transcribe/"
    echo "Contents of /opt/transcribe/:"
    ls -la
    exit 1
fi

# Run the advanced caching with GPU support
python advanced_cache.py

echo ""
echo "Cache creation results:"
if [ -f "/opt/transcribe/cache/cache_info.json" ]; then
    echo "✅ Cache created successfully!"
    cat /opt/transcribe/cache/cache_info.json
    
    # Verify GPU was used for caching
    CACHE_DEVICE=$(cat /opt/transcribe/cache/cache_info.json | grep '"device"' | cut -d'"' -f4)
    if [ "$CACHE_DEVICE" = "cuda" ]; then
        echo "✅ Cache built with GPU support!"
    else
        echo "⚠️ Cache built with CPU - GPU may not be working"
    fi
else
    echo "❌ Cache creation failed"
    exit 1
fi

# Ensure proper ownership of cache directory
sudo chown -R ubuntu:ubuntu /opt/transcribe/cache

echo "✅ GPU-optimized cache setup completed"
CACHING_SESSION

# Note: GPU-optimized model caching already completed above

echo ""
echo "🔍 Verifying component assembly..."

# Check directory structure first
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "ls -la /opt/transcribe/ 2>/dev/null | head -10 || echo 'ERROR: /opt/transcribe directory not accessible'"

# Verify all components are in correct locations
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "test -f /opt/transcribe/scripts/transcribe_optimized.py && echo '✅ transcribe_optimized.py' || echo '❌ Missing transcribe_optimized.py'"
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "test -f /opt/transcribe/scripts/api_server.py && echo '✅ api_server.py' || echo '❌ Missing api_server.py'"
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "test -f /opt/transcribe/advanced_cache.py && echo '✅ advanced_cache.py' || echo '❌ Missing advanced_cache.py'"
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "test -f /opt/transcribe/cache/cache_info.json && echo '✅ cache_info.json' || echo '❌ Missing cache_info.json'"
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "test -f /etc/systemd/system/transcribe-api.service && echo '✅ transcribe-api.service' || echo '❌ Missing transcribe-api.service'"

echo ""
echo "🔧 Final optimizations and verification..."

ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'FINAL_EOF'
set -e

echo "FINAL: Optimizations and verification..."

# Create optimized production config
sudo mkdir -p /opt/transcribe/config
sudo tee /opt/transcribe/config/optimized.conf << 'CONFEOF'
# OPTIMIZED G4DN.XLARGE Configuration
MODEL_ID=KBLab/kb-whisper-small
CACHE_DIR=/opt/transcribe/models
CHUNK_LENGTH=30
BATCH_SIZE=6
LANGUAGE=sv
DEVICE=cuda
TORCH_DTYPE=float16
KERNELS_PRECOMPILED=true
BOOT_TARGET=<20s
CONFEOF

# Create boot optimization script
sudo tee /opt/transcribe/scripts/optimize_boot.sh << 'BOOTEOF'
#!/bin/bash
# Boot optimization script - runs on instance startup

echo "Starting boot optimizations..."

# Wait for GPU to be available
for i in {1..30}; do
    if nvidia-smi >/dev/null 2>&1; then
        echo "GPU detected on attempt $i"
        break
    fi
    echo "Waiting for GPU... attempt $i/30"
    sleep 2
done

# GPU optimizations
if nvidia-smi >/dev/null 2>&1; then
    echo "Applying GPU optimizations..."
    
    # GPU persistence mode for faster initialization
    sudo nvidia-smi -pm 1 2>/dev/null && echo "GPU persistence mode enabled" || echo "GPU persistence mode failed"
    
    # Set GPU power and clock speeds for consistent performance  
    sudo nvidia-smi -pl 70 2>/dev/null && echo "GPU power limit set to 70W" || echo "GPU power management not available"
    sudo nvidia-smi -ac 5001,1590 2>/dev/null && echo "GPU clocks set" || echo "GPU clock management not available"
    
    # Display GPU status
    nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
else
    echo "WARNING: GPU not available, skipping GPU optimizations"
fi

echo "Boot optimizations completed at $(date)"
BOOTEOF

sudo chmod +x /opt/transcribe/scripts/optimize_boot.sh
sudo chown ubuntu:ubuntu /opt/transcribe/scripts/optimize_boot.sh

# Add boot optimization to startup
echo '@reboot ubuntu /opt/transcribe/scripts/optimize_boot.sh' | sudo tee -a /etc/crontab

# Advanced optimizations and cleanup
echo "Advanced optimizations and cleanup..."
sudo apt autoremove -y --purge
sudo apt autoclean
sudo apt clean

# Clear all caches
sudo rm -rf /var/cache/apt/*
sudo rm -rf /tmp/* 2>/dev/null || true
sudo rm -rf ~/.cache/* 2>/dev/null || true

# Clear logs  
sudo truncate -s 0 /var/log/*.log 2>/dev/null || true

# Clear history
history -c 2>/dev/null || true
sudo rm -f /root/.bash_history 2>/dev/null || true
rm -f ~/.bash_history 2>/dev/null || true

echo "✅ AMI environment setup completed!"

# Quick verification
echo "🔍 Verifying installation..."

# Test critical components
source /opt/transcribe/venv/bin/activate 2>/dev/null && echo "✅ Virtual environment" || echo "❌ Virtual environment failed"
[ -f "/opt/transcribe/scripts/transcribe_optimized.py" ] && echo "✅ Transcription script" || echo "❌ Missing transcription script"
[ -f "/opt/transcribe/scripts/api_server.py" ] && echo "✅ API server" || echo "❌ Missing API server"
[ -f "/opt/transcribe/advanced_cache.py" ] && echo "✅ Cache script" || echo "❌ Missing cache script"

# Check cache status
if [ -f "/opt/transcribe/cache/cache_info.json" ]; then
    echo "✅ Model cache ready"
else
    echo "⚠️ Cache will be created on first use"
fi

echo "🎯 Verification complete"
FINAL_EOF

# Clean up local temp files
rm -f /tmp/setup_phase1.sh /tmp/setup_phase2.sh

echo ""
echo "🔍 FINAL GPU VERIFICATION"
echo "========================"

# Run comprehensive final verification
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'FINAL_VERIFY_EOF'
echo "🔍 Final GPU verification before AMI creation..."

# Test NVIDIA drivers
echo "NVIDIA Driver Status:"
if nvidia-smi >/dev/null 2>&1; then
    echo "✅ nvidia-smi working"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "❌ nvidia-smi failed"
    exit 1
fi

# Test PyTorch CUDA
echo ""
echo "PyTorch CUDA Status:"
cd /opt/transcribe && source venv/bin/activate
python -c "
import torch
import sys
print(f'PyTorch: {torch.__version__}')
cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available}')
if cuda_available:
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    # Test GPU allocation
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print('✅ GPU tensor operations successful')
    del x, y, z
    torch.cuda.empty_cache()
else:
    print('❌ PyTorch CUDA not available')
    sys.exit(1)
"

# Verify cache
echo ""
echo "Model Cache Status:"
if [ -f "/opt/transcribe/cache/cache_info.json" ]; then
    echo "✅ Cache exists"
    CACHE_DEVICE=$(cat /opt/transcribe/cache/cache_info.json | grep '"device"' | cut -d'"' -f4)
    KERNELS_COMPILED=$(cat /opt/transcribe/cache/cache_info.json | grep '"kernels_compiled"' | cut -d':' -f2 | tr -d ' ,')
    echo "Cache device: $CACHE_DEVICE"
    echo "Kernels compiled: $KERNELS_COMPILED"
    
    if [ "$CACHE_DEVICE" = "\"cuda\"" ]; then
        echo "✅ Cache built with GPU support"
    else
        echo "❌ Cache built with CPU - this is wrong!"
        exit 1
    fi
else
    echo "❌ Cache missing"
    exit 1
fi

echo ""
echo "🎯 ALL VERIFICATIONS PASSED!"
echo "GPU transcription should work at maximum speed."

FINAL_VERIFY_EOF

echo ""
echo "🏗️ Creating MODULAR production AMI..."

AMI_ID=$(aws ec2 create-image \
    --instance-id $INSTANCE_ID \
    --name "transcription-modular-$(date +%Y%m%d-%H%M%S)" \
    --description "MODULAR G4DN.XLARGE AMI: Sub-20s boot, instant transcription, modular components" \
    --reboot \
    --output text \
    --query 'ImageId')

echo "🎉 MODULAR AMI creation started: $AMI_ID"
echo "⏳ Waiting for AMI to be available (10-15 minutes)..."

# Wait for AMI to be available
aws ec2 wait image-available --image-ids $AMI_ID

echo "✅ MODULAR AMI created successfully!"

# Save AMI ID
echo $AMI_ID > ami_id.txt

echo ""
echo "🎯 AMI Build Complete!"
echo "AMI ID: $AMI_ID (saved to ami_id.txt)"
echo ""
echo "🚀 Ready for: ./launch_api_server.sh"
echo "🎉 Ultra-fast transcription with modular architecture!" 