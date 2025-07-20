#!/bin/bash

# Quick Remote Instance Diagnostic
# Usage: ./check_remote_instance.sh [instance_ip]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <instance_ip>"
    echo "Example: $0 13.51.234.567"
    exit 1
fi

PUBLIC_IP="$1"

echo "🔍 Remote Instance Diagnostic"
echo "Instance IP: $PUBLIC_IP"
echo "==============================="

# Test SSH connectivity
echo "Testing SSH connectivity..."
if ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$PUBLIC_IP "echo 'SSH OK'" >/dev/null 2>&1; then
    echo "✅ SSH connection working"
else
    echo "❌ SSH connection failed"
    exit 1
fi

# Upload and run diagnostic
echo "Uploading diagnostic script..."
scp -i transcription-ec2.pem -o StrictHostKeyChecking=no diagnostic_check.py ubuntu@$PUBLIC_IP:/tmp/ >/dev/null 2>&1

echo "Running remote diagnostics..."
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'REMOTE_EOF'

echo "🔍 REMOTE DIAGNOSTIC RESULTS"
echo "============================="

# Check instance type
echo "Instance type:"
curl -s --connect-timeout 3 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "Unknown"

# Check GPU
echo ""
echo "GPU Status:"
if nvidia-smi >/dev/null 2>&1; then
    echo "✅ nvidia-smi working"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
else
    echo "❌ nvidia-smi not working"
fi

# Check PyTorch
echo ""
echo "PyTorch Status:"
cd /opt/transcribe 2>/dev/null && source venv/bin/activate 2>/dev/null
python3 -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__}')
    print(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch error: {e}')
" 2>/dev/null

# Check transcription environment
echo ""
echo "Environment Status:"
if [ -d "/opt/transcribe" ]; then
    echo "✅ /opt/transcribe exists"
    if [ -f "/opt/transcribe/cache/cache_info.json" ]; then
        echo "✅ Cache config exists"
        echo "Cache info:"
        cat /opt/transcribe/cache/cache_info.json 2>/dev/null | head -10
    else
        echo "❌ Cache config missing"
    fi
    
    if [ -d "/opt/transcribe/models" ]; then
        model_count=$(ls /opt/transcribe/models/ 2>/dev/null | wc -l)
        echo "✅ Models directory exists ($model_count files)"
    else
        echo "❌ Models directory missing"
    fi
else
    echo "❌ /opt/transcribe missing"
fi

# Check API service
echo ""
echo "Service Status:"
if systemctl is-active transcribe-api >/dev/null 2>&1; then
    echo "✅ transcribe-api service running"
else
    echo "❌ transcribe-api service not running"
fi

# Run comprehensive diagnostic if available
echo ""
echo "Running comprehensive diagnostic..."
if [ -f "/tmp/diagnostic_check.py" ]; then
    cd /opt/transcribe 2>/dev/null && source venv/bin/activate 2>/dev/null
    python /tmp/diagnostic_check.py 2>/dev/null || echo "Diagnostic script failed"
else
    echo "❌ Diagnostic script not available"
fi

REMOTE_EOF

echo ""
echo "==============================="
echo "🎯 Remote diagnostic complete!" 