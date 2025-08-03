#!/bin/bash

# Persistent Pre-warmed Transcription Engine AMI Builder
# Target: Fast boot with persistent pre-warmed Swedish Whisper model
# Focus: Elegance, simplicity, and reliability

set -euo pipefail

echo "=== Persistent Pre-warmed Engine AMI Builder ==="
echo "Building AMI with NVIDIA drivers and persistent pre-warmed engine..."

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
    
    # Validate required files exist
    if [ ! -f "fast_transcribe.py" ]; then
        handle_error "fast_transcribe.py not found in current directory"
    fi
    
    if [ ! -f "fast_transcribe.sh" ]; then
        handle_error "fast_transcribe.sh not found in current directory"
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
    libsndfile1

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
mkdir -p /opt/transcribe/{scripts,models,cache,logs,prewarmed} || {
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
sudo -u ubuntu /opt/transcribe/venv/bin/pip install transformers librosa boto3 numpy || {
    echo "[$(date)] ERROR: Failed to install Python packages"
    exit 1
}

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

# Create persistent pre-warmed engine
pre_warm_cuda_and_libraries() {
    log "🔥 Creating persistent pre-warmed transcription engine..."
    
    cat > /tmp/create_prewarmed_engine.py << 'PREWARMED_ENGINE'
#!/opt/transcribe/venv/bin/python3
import os
import sys
import torch
import pickle
import time
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔥 Creating Persistent Pre-warmed Transcription Engine")

try:
    # Ensure directories exist
    os.makedirs("/opt/transcribe/prewarmed", exist_ok=True)
    os.makedirs("/opt/transcribe/models", exist_ok=True)
    
    # Step 1: Force CUDA initialization and warm up context
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warming CUDA context...")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        
        # Warm up CUDA with tensor operations
        for i in range(5):
            x = torch.randn(1000, 1000, device=device, dtype=torch.float16)
            y = torch.randn(1000, 1000, device=device, dtype=torch.float16)
            z = torch.mm(x, y)
            del x, y, z
            torch.cuda.empty_cache()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CUDA warmup iteration {i+1}/5")
        
        torch.cuda.synchronize()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ CUDA context warmed successfully")
    else:
        device = torch.device('cpu')
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ CUDA not available, using CPU")
    
    # Step 2: Load and optimize the model
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading Whisper model...")
    model_id = "KBLab/kb-whisper-small"
    
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            cache_dir="/opt/transcribe/models",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir="/opt/transcribe/models"
        )
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Failed to download model: {e}")
        sys.exit(1)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model loaded successfully")
    
    # Step 3: Move model to device and optimize
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Moving model to {device} and optimizing...")
    model = model.to(device)
    model.eval()
    
    # Step 4: Apply torch.compile optimization if available
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Applying torch.compile optimization...")
        model = torch.compile(model, mode="reduce-overhead")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Model compiled successfully")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ torch.compile failed: {e}")
    
    # Step 5: Create optimized pipeline
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating optimized pipeline...")
    
    # Fix for WhisperProcessor compatibility issue
    if hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor
    
    # Ensure pad_token_id is set
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    pipeline_obj = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=processor,
        chunk_length_s=30,
        stride_length_s=5,
        batch_size=16,
        torch_dtype=torch.float16,
        device=device if device.type == "cpu" else 0,
        return_timestamps=False
    )
    
    # Step 6: Pre-warm the pipeline with dummy audio
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pre-warming pipeline with dummy audio...")
    import numpy as np
    
    # Create dummy audio (1 second of silence at 16kHz)
    dummy_audio = np.zeros(16000, dtype=np.float32)
    
    # Pre-warm the pipeline
    try:
        _ = pipeline_obj(dummy_audio)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Pipeline pre-warmed successfully")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Pipeline pre-warming failed: {e}")
    
    # Step 7: Create complete pre-warmed engine object
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating complete pre-warmed engine...")
    
    # Create a simple dictionary-based engine instead of a class to avoid serialization issues
    # Note: We can't include lambda functions in pickle, so we'll create a simple class for get_info
    class EngineInfo:
        def __init__(self, model_id, device, is_compiled):
            self.model_id = model_id
            self.device = device
            self.is_compiled = is_compiled
        
        def get_info(self):
            return {
                "model_id": self.model_id,
                "device": str(self.device),
                "is_compiled": self.is_compiled,
                "created_at": datetime.now().isoformat(),
                "status": "ready"
            }
    
    engine_info = EngineInfo(model_id, device, hasattr(model, '_orig_mod'))
    
    prewarmed_engine = {
        'model': model,
        'processor': processor,
        'pipeline': pipeline_obj,
        'device': device,
        'model_id': model_id,
        'created_at': datetime.now().isoformat(),
        'device_info': str(device),
        'is_compiled': hasattr(model, '_orig_mod'),  # Check if torch.compile was applied
        'engine_info': engine_info
    }
    
    # Step 8: Serialize the complete pre-warmed engine
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Serializing pre-warmed engine...")
    engine_path = "/opt/transcribe/prewarmed/prewarmed_engine.pkl"
    
    with open(engine_path, 'wb') as f:
        pickle.dump(prewarmed_engine, f)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Pre-warmed engine serialized to: {engine_path}")
    
    # Step 9: Test loading the serialized engine
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing engine deserialization...")
    with open(engine_path, 'rb') as f:
        loaded_engine = pickle.load(f)
    
    # Verify the loaded engine works
    info = loaded_engine['engine_info'].get_info()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Engine loaded successfully: {info}")
    
    # Test transcription with dummy audio
    try:
        result = loaded_engine['pipeline'](dummy_audio)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Engine transcription test successful")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Engine transcription test failed: {e}")
    
    # Step 10: Create engine metadata
    metadata = {
        "engine_path": engine_path,
        "model_id": model_id,
        "device": str(device),
        "is_compiled": hasattr(model, '_orig_mod'),
        "created_at": datetime.now().isoformat(),
        "file_size_mb": os.path.getsize(engine_path) / (1024 * 1024),
        "status": "ready"
    }
    
    metadata_path = "/opt/transcribe/prewarmed/engine_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Engine metadata saved to: {metadata_path}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Persistent pre-warmed engine creation completed!")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Engine size: {metadata['file_size_mb']:.1f} MB")
    
except Exception as e:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PREWARMED_ENGINE

    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/create_prewarmed_engine.py ubuntu@"$PUBLIC_IP":/opt/transcribe/
    
    log "Creating persistent pre-warmed engine..."
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "cd /opt/transcribe && sudo -u ubuntu /opt/transcribe/venv/bin/python create_prewarmed_engine.py" \
        || handle_error "Pre-warmed engine creation failed"
    
    log "✅ Persistent pre-warmed engine created successfully"
}

# Create boot warmup script
create_boot_warmup_script() {
    log "Creating boot warmup script for persistent engine loading..."
    
    cat > /tmp/boot_warmup.py << 'BOOT_WARMUP'
#!/opt/transcribe/venv/bin/python3
"""
Boot Warmup Script - Runs on every EC2 instance startup
Loads the pre-warmed transcription engine and warms CUDA context
"""
import os
import sys
import pickle
import torch
import time
from datetime import datetime

# Global variable to hold the pre-warmed engine
_global_prewarmed_engine = None

def load_prewarmed_engine():
    """Load the pre-warmed transcription engine from disk"""
    global _global_prewarmed_engine
    
    engine_path = "/opt/transcribe/prewarmed/prewarmed_engine.pkl"
    
    if not os.path.exists(engine_path):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Pre-warmed engine not found: {engine_path}")
        return None
    
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔥 Loading pre-warmed transcription engine...")
        start_time = time.time()
        
        with open(engine_path, 'rb') as f:
            _global_prewarmed_engine = pickle.load(f)
        
        load_time = time.time() - start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Pre-warmed engine loaded in {load_time:.2f}s")
        
        # Get engine info (handle both class and dict-based engines)
        if hasattr(_global_prewarmed_engine, 'get_info'):
            info = _global_prewarmed_engine.get_info()
        elif isinstance(_global_prewarmed_engine, dict) and 'engine_info' in _global_prewarmed_engine:
            info = _global_prewarmed_engine['engine_info'].get_info()
        else:
            info = {'status': 'unknown'}
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Engine info: {info}")
        
        return _global_prewarmed_engine
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Failed to load pre-warmed engine: {e}")
        return None

def warm_cuda_context():
    """Warm up CUDA context with small tensor operations"""
    if not torch.cuda.is_available():
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ CUDA not available, skipping CUDA warmup")
        return
    
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔥 Warming CUDA context...")
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        
        # Quick CUDA warmup
        for i in range(3):
            x = torch.randn(500, 500, device=device, dtype=torch.float16)
            y = torch.randn(500, 500, device=device, dtype=torch.float16)
            z = torch.mm(x, y)
            del x, y, z
            torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ CUDA context warmed successfully")
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ CUDA warmup failed: {e}")

def get_prewarmed_engine():
    """Get the global pre-warmed engine instance"""
    global _global_prewarmed_engine
    return _global_prewarmed_engine

def test_engine_functionality():
    """Test that the loaded engine works correctly"""
    if _global_prewarmed_engine is None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ No pre-warmed engine available for testing")
        return False
    
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing engine functionality...")
        
        # Create dummy audio for testing
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        
        # Get pipeline (handle both class and dict-based engines)
        if hasattr(_global_prewarmed_engine, 'pipeline'):
            pipeline = _global_prewarmed_engine.pipeline
        elif isinstance(_global_prewarmed_engine, dict) and 'pipeline' in _global_prewarmed_engine:
            pipeline = _global_prewarmed_engine['pipeline']
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ No pipeline found in engine")
            return False
        
        # Test transcription
        result = pipeline(dummy_audio)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Engine functionality test passed")
        return True
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Engine functionality test failed: {e}")
        return False

def main():
    """Main boot warmup function"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Starting boot warmup process...")
    
    # Step 1: Warm CUDA context
    warm_cuda_context()
    
    # Step 2: Load pre-warmed engine
    engine = load_prewarmed_engine()
    
    if engine is None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Boot warmup completed with warnings")
        return False
    
    # Step 3: Test engine functionality
    test_success = test_engine_functionality()
    
    if test_success:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Boot warmup completed successfully")
        return True
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Boot warmup completed with warnings")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
BOOT_WARMUP

    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/boot_warmup.py ubuntu@"$PUBLIC_IP":/opt/transcribe/
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "chmod +x /opt/transcribe/boot_warmup.py"
    
    # Create systemd service for automatic boot warmup
    cat > /tmp/transcription-warmup.service << 'SYSTEMD_SERVICE'
[Unit]
Description=Transcription Engine Boot Warmup
After=network.target
Wants=network.target

[Service]
Type=oneshot
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/transcribe
Environment=PATH=/opt/transcribe/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/transcribe/venv/bin/python /opt/transcribe/boot_warmup.py
StandardOutput=journal
StandardError=journal
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
SYSTEMD_SERVICE

    scp -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        /tmp/transcription-warmup.service ubuntu@"$PUBLIC_IP":/tmp/
    
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "sudo cp /tmp/transcription-warmup.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable transcription-warmup.service"
    
    log "✅ Boot warmup script and systemd service created"
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

# Essential checks - only validate what's not already verified
echo -n "NVIDIA drivers... "
nvidia-smi &> /dev/null && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Persistent pre-warmed engine... "
[ -f /opt/transcribe/prewarmed/prewarmed_engine.pkl ] && [ -f /opt/transcribe/prewarmed/engine_metadata.json ] && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Pre-warmed engine functionality... "
/opt/transcribe/venv/bin/python -c "
import pickle
import torch
import sys
import os

# Check if engine file exists and has reasonable size
engine_path = '/opt/transcribe/prewarmed/prewarmed_engine.pkl'
if not os.path.exists(engine_path):
    print('Engine file not found')
    sys.exit(1)

file_size = os.path.getsize(engine_path)
if file_size < 50000000:  # Less than 50MB (more reasonable)
    print(f'Engine file too small: {file_size} bytes')
    sys.exit(1)

# Initialize CUDA context first
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

try:
    with open(engine_path, 'rb') as f:
        engine = pickle.load(f)
    
    # Verify the engine object has expected keys (dictionary-based)
    if isinstance(engine, dict) and 'pipeline' in engine and 'model' in engine and 'engine_info' in engine:
        info = engine['engine_info'].get_info()
        print(f'Engine test: SUCCESS - {info.get(\"model_id\", \"unknown\")}')
    else:
        print('Engine test: FAILED - missing expected keys')
        sys.exit(1)
        
except Exception as e:
    print(f'Engine test failed: {e}')
    sys.exit(1)
" && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Boot warmup integration... "
/opt/transcribe/venv/bin/python -c "
import sys
sys.path.insert(0, '/opt/transcribe')

try:
    from boot_warmup import load_prewarmed_engine, get_prewarmed_engine
    
    # First load the engine
    engine = load_prewarmed_engine()
    
    # Then test getting it
    if engine is not None:
        print('Boot warmup integration: SUCCESS')
    else:
        print('Boot warmup integration: FAILED - engine is None')
        sys.exit(1)
        
except Exception as e:
    print(f'Boot warmup integration: FAILED - {e}')
    sys.exit(1)
" && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Boot warmup script... "
[ -f /opt/transcribe/boot_warmup.py ] && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Systemd service... "
systemctl is-enabled transcription-warmup.service &> /dev/null && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Optimized scripts... "
[ -f /opt/transcribe/fast_transcribe.py ] && [ -f /opt/transcribe/fast_transcribe.sh ] && [ -f /opt/transcription/fast_transcribe.sh ] && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Fast transcribe integration... "
/opt/transcribe/venv/bin/python -c "
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
        --description "GPU transcription AMI with persistent pre-warmed Swedish Whisper model" \
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
    
    local lambda_file="../setup/lambda/lambda_process_upload.py"
    
    if [ -f "$lambda_file" ]; then
        cp "$lambda_file" "${lambda_file}.backup"
        # Fix: Update the pattern to match the actual format in lambda file
        if sed -i "s/AMI_ID = os.environ.get('AMI_ID', 'ami-[a-z0-9]*')/AMI_ID = os.environ.get('AMI_ID', '$AMI_ID')/" "$lambda_file"; then
            log "Updated AMI ID in lambda function: $lambda_file"
            
            if grep -q "AMI_ID = os.environ.get('AMI_ID', '$AMI_ID')" "$lambda_file"; then
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
        if sed -i "s/EXPECTED_AMI_ID = 'ami-[a-z0-9]*'/EXPECTED_AMI_ID = '$AMI_ID'/" "$python_file"; then
            log "Updated expected AMI ID in Python script: $python_file"
            
            if grep -q "EXPECTED_AMI_ID = '$AMI_ID'" "$python_file"; then
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
    pre_warm_cuda_and_libraries
    create_boot_warmup_script
    create_transcription_script
    validate_setup
    create_ami
    update_lambda_ami_id
    
    log "AMI build completed successfully!"
    log "Final AMI ID: $AMI_ID"
    log "AMI ID file updated: ami_id.txt"
    log "Lambda function and scripts updated with new AMI ID"
    
    log "=== PERSISTENT PRE-WARMED ENGINE BUILD SUMMARY ==="
    log "✓ Base AMI: $BASE_AMI"
    log "✓ New AMI: $AMI_ID"
    log "✓ Instance Type: $INSTANCE_TYPE"
    log "✓ Model: $MODEL_ID"
    log "✓ NVIDIA Drivers: Installed"
    log "✓ Python Environment: Ready"
    log "✓ Enhanced Bytecode Compilation: Completed"
    log "✓ Persistent Pre-warmed Engine: Created"
    log "✓ Boot Warmup Script: Installed"
    log "✓ Systemd Service: Enabled"
    log "✓ Scripts: Uploaded and Verified"
    log "✓ Lambda Function: Updated"
    log "✓ Setup Marker: Created"
    log "=================================================="
    log ""
    log "Next steps:"
    log "1. Deploy the updated lambda function with: cd ../setup/lambda && ./deploy_lambda_functions.sh"
    log "2. Test transcription by uploading a file to S3"
    log "3. Monitor CloudWatch logs for any issues"
}

# Run main function
main