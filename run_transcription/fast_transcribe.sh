#!/bin/bash

# Fast GPU Transcription - EC2 Startup Script
# ============================================
# Startup script for EC2 instances launched by Lambda
# Downloads audio from S3, transcribes, uploads results

set -euo pipefail

# Configuration
REGION="eu-north-1"
S3_BUCKET="transcription-curevo"
DYNAMODB_TABLE="transcription-jobs"

# Enhanced logging function with DynamoDB status updates
log_msg() {
    local message="$1"
    local status_level="${2:-INFO}"
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$status_level] $message"
    
    # Update DynamoDB with status if this is a major milestone
    if [[ "$status_level" == "STATUS" ]]; then
        aws dynamodb update-item \
            --table-name "$DYNAMODB_TABLE" \
            --key "{\"job_id\": {\"S\": \"$JOB_ID\"}}" \
            --update-expression "SET status_message = :message, updated_at = :updated_at" \
            --expression-attribute-values "{\":message\": {\"S\": \"$message\"}, \":updated_at\": {\"S\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}}" \
            --region "$REGION" >/dev/null 2>&1 || true
    fi
}

log_msg "=== Fast GPU Transcription Startup Script ===" "STATUS"

# Get job information from instance tags
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
log_msg "Instance ID: $INSTANCE_ID"

# Get job_id from instance tags
JOB_ID=$(aws ec2 describe-tags \
    --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=JobId" \
    --query 'Tags[0].Value' --output text --region "$REGION" 2>/dev/null || echo "")

if [[ -z "$JOB_ID" ]]; then
    log_msg "Error: JobId not found in instance tags"
    exit 1
fi

log_msg "Job ID: $JOB_ID"

# Get S3 key from DynamoDB
S3_KEY=$(aws dynamodb get-item \
    --table-name "$DYNAMODB_TABLE" \
    --key "{\"job_id\": {\"S\": \"$JOB_ID\"}}" \
    --query 'Item.s3_key.S' --output text --region "$REGION" 2>/dev/null || echo "")

if [[ -z "$S3_KEY" ]]; then
    log_msg "Error: S3 key not found in DynamoDB for job $JOB_ID"
    exit 1
fi

log_msg "S3 Key: $S3_KEY"

# Update DynamoDB status to processing
aws dynamodb update-item \
    --table-name "$DYNAMODB_TABLE" \
    --key "{\"job_id\": {\"S\": \"$JOB_ID\"}}" \
    --update-expression "SET #status = :status, updated_at = :updated_at" \
    --expression-attribute-names '{"#status": "status"}' \
    --expression-attribute-values "{\":status\": {\"S\": \"processing\"}, \":updated_at\": {\"S\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}}" \
    --region "$REGION"

log_msg "Status updated to processing" "STATUS"

# Wait for instance to be fully ready
log_msg "Waiting for instance to be fully ready..."
sleep 30

# Update system packages
log_msg "Updating system packages..."
apt-get update -y > /dev/null 2>&1 || log_msg "Warning: Package update failed"

# Install required packages
log_msg "Installing required packages..."
apt-get install -y awscli python3-pip python3-venv ffmpeg > /dev/null 2>&1 || log_msg "Warning: Package installation failed"

# Create transcription directory
log_msg "Creating transcription directory..."
mkdir -p /opt/transcription
cd /opt/transcription

# Determine file extension from S3 key
log_msg "Determining file extension from S3 key: $S3_KEY"
FILE_EXTENSION="${S3_KEY##*.}"
if [[ "$FILE_EXTENSION" == "$S3_KEY" ]]; then
    # No extension found, assume mp3
    FILE_EXTENSION="mp3"
    log_msg "Warning: No file extension detected, assuming .mp3"
fi

# Validate file extension
VALID_EXTENSIONS=("mp3" "wav" "m4a" "flac" "ogg" "aac" "wma")
EXTENSION_VALID=false
for ext in "${VALID_EXTENSIONS[@]}"; do
    if [[ "$FILE_EXTENSION" == "$ext" ]]; then
        EXTENSION_VALID=true
        break
    fi
done

if [[ "$EXTENSION_VALID" == false ]]; then
    log_msg "Error: Unsupported file extension: $FILE_EXTENSION"
    exit 1
fi

log_msg "Using file extension: $FILE_EXTENSION"

# Download audio file from S3
log_msg "Downloading audio file from S3..." "STATUS"
if ! aws s3 cp "s3://$S3_BUCKET/$S3_KEY" "./audio_file.$FILE_EXTENSION" --region "$REGION"; then
    log_msg "Error: Failed to download audio file from S3"
    exit 1
fi

# Verify downloaded file
if [[ ! -f "./audio_file.$FILE_EXTENSION" ]]; then
    log_msg "Error: Downloaded file not found: ./audio_file.$FILE_EXTENSION"
    exit 1
fi

FILE_SIZE=$(stat -c%s "./audio_file.$FILE_EXTENSION" 2>/dev/null || echo "0")
log_msg "Downloaded file size: $FILE_SIZE bytes"

if [[ "$FILE_SIZE" -eq 0 ]]; then
    log_msg "Error: Downloaded file is empty"
    exit 1
fi

# Check if AMI setup is complete
if [[ ! -f "/opt/transcribe/.setup_complete" ]]; then
    log_msg "Error: AMI setup incomplete - missing .setup_complete marker"
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

# Copy audio file to transcribe directory
cp /opt/transcription/audio_file.$FILE_EXTENSION ./
log_msg "Audio file copied to transcribe directory"

# Run transcription
log_msg "=== Starting Transcription ===" "STATUS"
START_TIME=$(date +%s)

if python3 fast_transcribe.py "audio_file.$FILE_EXTENSION"; then
    log_msg "Transcription completed successfully!" "STATUS"
    TRANSCRIPTION_SUCCESS=true
else
    log_msg "Error: Transcription failed" "STATUS"
    TRANSCRIPTION_SUCCESS=false
fi

# Show GPU memory usage after transcription
log_msg "Final GPU memory status:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | head -1 | while read used free; do
    log_msg "  Used: ${used}MB, Free: ${free}MB"
done

# Upload results to S3
log_msg "Uploading results to S3..." "STATUS"
if ls transcription_*.txt 1> /dev/null 2>&1; then
    if aws s3 cp transcription_*.txt "s3://$S3_BUCKET/results/$JOB_ID/" --region "$REGION"; then
        log_msg "Results uploaded to S3: s3://$S3_BUCKET/results/$JOB_ID/"
    else
        log_msg "Warning: Failed to upload results to S3"
    fi
else
    log_msg "Warning: No result files found to upload"
fi

# Get callback info from DynamoDB
log_msg "Getting callback information..."
CALLBACK_INFO=$(aws dynamodb get-item \
    --table-name "$DYNAMODB_TABLE" \
    --key "{\"job_id\": {\"S\": \"$JOB_ID\"}}" \
    --query 'Item.callback_url.S' \
    --output text \
    --region "$REGION" 2>/dev/null || echo "")

CALLBACK_SECRET=$(aws dynamodb get-item \
    --table-name "$DYNAMODB_TABLE" \
    --key "{\"job_id\": {\"S\": \"$JOB_ID\"}}" \
    --query 'Item.callback_secret.S' \
    --output text \
    --region "$REGION" 2>/dev/null || echo "")

# Update DynamoDB status based on transcription result
log_msg "Updating job status..."
if [[ "$TRANSCRIPTION_SUCCESS" == "true" ]]; then
    FINAL_STATUS="completed"
else
    FINAL_STATUS="failed"
fi

aws dynamodb update-item \
    --table-name "$DYNAMODB_TABLE" \
    --key "{\"job_id\": {\"S\": \"$JOB_ID\"}}" \
    --update-expression "SET #status = :status, updated_at = :updated_at" \
    --expression-attribute-names '{"#status": "status"}' \
    --expression-attribute-values "{\":status\": {\"S\": \"$FINAL_STATUS\"}, \":updated_at\": {\"S\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}}" \
    --region "$REGION"

# Send webhook callback if URL is available
if [[ -n "$CALLBACK_INFO" && "$CALLBACK_INFO" != "None" ]]; then
    log_msg "Sending webhook callback to: $CALLBACK_INFO"
    
    # Prepare webhook payload based on transcription success
    if [[ "$TRANSCRIPTION_SUCCESS" == "true" ]]; then
        # Success case - include transcription results
        TRANSCRIPTION_TEXT=""
        if ls transcription_*.txt 1> /dev/null 2>&1; then
            # Get the latest transcription file
            LATEST_FILE=$(ls -t transcription_*.txt | head -1)
            if [[ -n "$LATEST_FILE" ]]; then
                # Extract just the transcription text (after "=== TRANSCRIPTION ===")
                TRANSCRIPTION_TEXT=$(sed -n '/=== TRANSCRIPTION ===/,$p' "$LATEST_FILE" | tail -n +2 | tr '\n' ' ' | sed 's/"/\\"/g')
            fi
        fi
        
        if [[ -z "$TRANSCRIPTION_TEXT" ]]; then
            TRANSCRIPTION_TEXT="Transcription completed but no text found"
        fi
        
        WEBHOOK_PAYLOAD="{
            \"job_id\": \"$JOB_ID\",
            \"status\": \"completed\",
            \"transcript\": {
                \"text\": \"$TRANSCRIPTION_TEXT\",
                \"segments\": []
            },
            \"metadata\": {
                \"processing_time\": \"$(($(date +%s) - START_TIME))s\",
                \"file_size\": \"$FILE_SIZE\",
                \"model\": \"kb-whisper-small\"
            }
        }"
    else
        # Failure case - include error information
        WEBHOOK_PAYLOAD="{
            \"job_id\": \"$JOB_ID\",
            \"status\": \"failed\",
            \"error\": \"Transcription process failed\",
            \"error_code\": \"TRANSCRIPTION_FAILED\",
            \"metadata\": {
                \"file_size\": \"$FILE_SIZE\",
                \"model\": \"kb-whisper-small\"
            }
        }"
    fi
    
    # Add HMAC signature if secret is available
    if [[ -n "$CALLBACK_SECRET" && "$CALLBACK_SECRET" != "None" ]]; then
        WEBHOOK_SIGNATURE=$(echo -n "$WEBHOOK_PAYLOAD" | openssl dgst -sha256 -hmac "$CALLBACK_SECRET" -binary | base64)
        WEBHOOK_HEADERS="-H 'X-Webhook-Signature: hmac-sha256=$WEBHOOK_SIGNATURE'"
    else
        WEBHOOK_HEADERS=""
    fi
    
    # Send webhook
    if curl -X POST "$CALLBACK_INFO" \
        -H 'Content-Type: application/json' \
        $WEBHOOK_HEADERS \
        -d "$WEBHOOK_PAYLOAD" \
        --max-time 30 \
        --connect-timeout 10 >/dev/null 2>&1; then
        log_msg "Webhook callback sent successfully"
    else
        log_msg "Warning: Failed to send webhook callback"
    fi
else
    log_msg "No callback URL configured, skipping webhook"
fi

# Show results summary
TOTAL_TIME=$(($(date +%s) - START_TIME))
log_msg ""
log_msg "=== Transcription Complete ==="
log_msg "Total time: ${TOTAL_TIME} seconds"
log_msg "Instance: $INSTANCE_ID"
log_msg "Job ID: $JOB_ID"

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
    fi
else
    log_msg "No result files generated"
fi

log_msg ""
log_msg "Fast transcription completed!"

# Terminate instance
log_msg "Terminating instance..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null 2>&1 || true 