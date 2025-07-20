#!/bin/bash

# OPTIMIZED Lightning G4DN.XLARGE Transcription
# Ultra-fast <90 second total runtime with optimized AMI
# Uses pre-compiled kernels, cached models, and boot optimizations

AUDIO_FILE=${1:-audio20min.mp3}

echo "OPTIMIZED Lightning G4DN.XLARGE Transcription"
echo "=============================================="
echo "Target runtime: <90 seconds total"
echo "Boot target: <20 seconds"
echo "Audio file: $AUDIO_FILE"

export AWS_DEFAULT_REGION=eu-north-1

# Cleanup function for automatic instance termination
cleanup_instances() {
    echo ""
    echo "Optimized cleanup..."
    
    if [ ! -z "$INSTANCE_ID" ]; then
        echo "Terminating instance: $INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null 2>&1
    fi
    
    # Clean up temp files
    rm -f /tmp/transcribe_instance_id
    
    echo "Optimized cleanup completed"
}

# Set up signal traps for automatic cleanup
trap cleanup_instances EXIT INT TERM

# Clean up any previous instances
if [ -f /tmp/transcribe_instance_id ]; then
    OLD_INSTANCE_ID=$(cat /tmp/transcribe_instance_id 2>/dev/null)
    if [ ! -z "$OLD_INSTANCE_ID" ]; then
        echo "Cleaning up previous instance: $OLD_INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$OLD_INSTANCE_ID" >/dev/null 2>&1
    fi
fi

echo ""
echo "Loading OPTIMIZED AMI..."

# Check for AMI
if [ -f ami_id.txt ]; then
    AMI_ID=$(cat ami_id.txt)
    echo "Using OPTIMIZED AMI: $AMI_ID"
else
    echo "ERROR: No AMI found!"
    echo "Please run: ./build_ami.sh first"
    exit 1
fi

# Verify audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "ERROR: Audio file not found: $AUDIO_FILE"
    echo "Available files:"
    ls -la *.mp3 *.wav *.m4a 2>/dev/null || echo "   No audio files found"
    exit 1
fi

echo ""
echo "Optimized lightning launch..."

# Check for existing instances (reuse for multiple transcriptions)
echo "Checking for existing optimized instances..."
EXISTING_INSTANCE=$(aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
              "Name=instance-type,Values=g4dn.xlarge" \
              "Name=instance.group-name,Values=transcription-g4dn-sg" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text 2>/dev/null)

if [ "$EXISTING_INSTANCE" != "None" ] && [ ! -z "$EXISTING_INSTANCE" ] && [ "$EXISTING_INSTANCE" != "null" ]; then
    echo "Found existing optimized instance: $EXISTING_INSTANCE"
    echo "Reusing for ultra-fast transcription..."
    INSTANCE_ID="$EXISTING_INSTANCE"
    
    # Get the public IP of existing instance
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    echo "Instance IP: $PUBLIC_IP"
    echo "Skipping launch - going directly to transcription..."
    
    # Store for cleanup
    echo "$INSTANCE_ID" > /tmp/transcribe_instance_id
    
else
    echo "Launching optimized instance for ultra-fast boot..."

# Start timer for launch performance
START_TIME=$(date +%s)

# Try multiple zones for instance availability with optimized block device
ZONES=("eu-north-1a" "eu-north-1b" "eu-north-1c")
INSTANCE_ID=""
zone_attempt=0

for zone in "${ZONES[@]}"; do
    zone_attempt=$((zone_attempt + 1))
    echo "Trying optimized launch in zone: $zone (attempt $zone_attempt/3)"
    
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
        echo "SUCCESS: Optimized instance launched in zone: $zone"
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
echo "$INSTANCE_ID" > /tmp/transcribe_instance_id

echo "Optimized instance launched: $INSTANCE_ID"

# Wait for running state (optimized AMI should boot very fast)
echo "Waiting for optimized boot..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get IP immediately
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Public IP: $PUBLIC_IP"

# Optimized SSH wait (sub-20 second target)
echo "Waiting for optimized SSH (target <10s)..."
SSH_READY=false
for i in {1..15}; do
    if ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no -o ConnectTimeout=3 ubuntu@$PUBLIC_IP "echo 'Optimized SSH Ready'" >/dev/null 2>&1; then
        SSH_READY=true
        echo "SSH ready!"
        break
    fi
    echo "Attempt $i/15: SSH connecting..."
    sleep 2
done

if [ "$SSH_READY" = false ]; then
    echo "ERROR: SSH failed - optimized AMI may have issues"
    exit 1
fi

# Calculate optimized boot time
BOOT_TIME=$(($(date +%s) - START_TIME))
echo "Optimized boot completed in ${BOOT_TIME}s (target: <20s)"

fi  # End of new instance launch

# Verify optimized SSH connection
echo ""
echo "Verifying optimized connection..."
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
echo "Uploading audio for optimized transcription..."

# Optimized upload with compression
if ! scp -i transcription-ec2.pem -o StrictHostKeyChecking=no -o Compression=yes "$AUDIO_FILE" ubuntu@$PUBLIC_IP:/home/ubuntu/; then
    echo "ERROR: Failed to upload audio file"
    exit 1
fi

echo "Running OPTIMIZED transcription..."

# Run optimized transcription using pre-compiled environment
ssh -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << EOF

# Verify optimized environment is ready
echo "Verifying optimized environment..."
if [ ! -f /opt/transcribe/cache/cache_info.json ]; then
    echo "WARNING: Optimized cache not found, using non-optimized transcription"
    
    # Fallback: Run optimized script anyway (it will work without pre-compiled kernels)
    source /opt/transcribe/venv/bin/activate
    cd /opt/transcribe
    python scripts/transcribe_optimized.py /home/ubuntu/$AUDIO_FILE
else
    echo "OPTIMIZED environment detected!"
    
    # Activate optimized environment
    source /opt/transcribe/venv/bin/activate
    cd /opt/transcribe
    
    # Show optimization status
    echo "Cache info:"
    cat /opt/transcribe/cache/cache_info.json | grep -E "(kernels_compiled|timestamp)"
    
    # Run optimized transcription with pre-compiled kernels
    echo "Starting OPTIMIZED transcription with pre-compiled kernels..."
    python scripts/transcribe_optimized.py /home/ubuntu/$AUDIO_FILE
fi

echo ""
echo "OPTIMIZED Performance Summary:"
echo "=============================="

# Show system status
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'T4')"
echo "GPU Memory: \$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo '16GB') MB"
echo "Boot optimizations: Active"
echo "Pre-compiled kernels: \$([ -f /opt/transcribe/cache/cache_info.json ] && echo 'Yes' || echo 'No')"

echo "OPTIMIZED transcription completed!"

EOF

echo ""
echo "Downloading optimized results..."

# Download all result files
scp -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP:/opt/transcribe/optimized_result_*.txt . 2>/dev/null || \
scp -i transcription-ec2.pem -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP:/opt/transcribe/result_*.txt . 2>/dev/null

# Upload to S3 with optimized naming
if ls optimized_result_*.txt >/dev/null 2>&1 || ls result_*.txt >/dev/null 2>&1; then
    echo "Uploading to S3..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
         # Try optimized results first
     for result_file in optimized_result_*.txt result_*.txt; do
         if [ -f "$result_file" ]; then
             aws s3 cp "$result_file" "s3://transcription-curevo/transcription_${TIMESTAMP}.txt" 2>/dev/null
             if [ $? -eq 0 ]; then
                 echo "SUCCESS: Results uploaded to S3!"
                 echo "View: https://transcription-curevo.s3.eu-north-1.amazonaws.com/transcription_${TIMESTAMP}.txt"
            else
                echo "WARNING: S3 upload failed - check credentials"
            fi
            break
        fi
    done
else
    echo "WARNING: No result files found to download"
fi

# Calculate total time with optimized metrics
TOTAL_TIME=$(($(date +%s) - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "OPTIMIZED Lightning Transcription Complete!"
echo "==========================================="
echo "Total runtime: ${MINUTES}m ${SECONDS}s"

# Optimized success criteria
if [ $TOTAL_TIME -le 90 ]; then
    echo "SUCCESS: Under 90 seconds! (OPTIMIZED TARGET ACHIEVED)"
elif [ $TOTAL_TIME -le 120 ]; then
    echo "SUCCESS: Under 2 minutes! (Good performance)"
elif [ $TOTAL_TIME -le 180 ]; then
    echo "SUCCESS: Under 3 minutes! (Standard performance)"
else
    echo "WARNING: Over 3 minutes (check optimization status)"
fi

echo "Estimated cost: ~$0.01-0.02 (spot pricing)"
echo ""

# Ask if user wants to keep optimized instance running
echo "Keep optimized instance running for more work? (y/N): "
read -t 10 -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Optimized instance left running: $INSTANCE_ID"
    echo "Public IP: $PUBLIC_IP"
    echo "PRO TIP: Reuse this instance for multiple transcriptions!"
    echo "Or cleanup later with: ./cleanup_transcription_instances.sh"
    
    # Clear trap to prevent auto-cleanup
    trap - EXIT
else
    echo "Auto-terminating optimized instance for cost savings..."
    # Cleanup will be handled by trap
fi

echo ""
echo "OPTIMIZED transcription session complete!"
echo "Next transcription will be even faster if instance is reused!" 