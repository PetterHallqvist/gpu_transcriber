#!/bin/bash

# Fast GPU Transcription - Ultra-Lean EC2 Startup Script
# ======================================================
# AMI-optimized: All dependencies pre-installed, minimal verification
# Goal: Start transcription within 10 seconds of instance launch

set -euo pipefail

# Configuration
REGION="eu-north-1"
S3_BUCKET="transcription-curevo"
DYNAMODB_TABLE="transcription-jobs"

# Enhanced logging function with timestamp
log_msg() {
    local message="$1"
    local status_level="${2:-INFO}"
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$status_level] $message"
    
    # Update DynamoDB with status if this is a major milestone and JOB_ID is set
    if [[ "$status_level" == "STATUS" ]] && [[ -n "${JOB_ID:-}" ]]; then
        aws dynamodb update-item \
            --table-name "$DYNAMODB_TABLE" \
            --key "{\"job_id\": {\"S\": \"$JOB_ID\"}}" \
            --update-expression "SET status_message = :message, updated_at = :updated_at" \
            --expression-attribute-values "{\":message\": {\"S\": \"$message\"}, \":updated_at\": {\"S\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}}" \
            --region "$REGION" >/dev/null 2>&1 || true
    fi
}

# Get instance metadata and AMI information
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
AMI_ID=$(curl -s http://169.254.169.254/latest/meta-data/ami-id 2>/dev/null || echo "unknown")
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")

log_msg "Instance ID: $INSTANCE_ID"
log_msg "AMI ID: $AMI_ID"
log_msg "Instance Type: $INSTANCE_TYPE"

# Get job_id from instance tags
JOB_ID=$(aws ec2 describe-tags \
    --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=JobId" \
    --query 'Tags[0].Value' \
    --output text \
    --region "$REGION" 2>/dev/null || echo "")

if [[ -z "$JOB_ID" ]] || [[ "$JOB_ID" == "None" ]]; then
    # Fallback: get job_id from DynamoDB
    JOB_ID=$(aws dynamodb scan \
        --table-name "$DYNAMODB_TABLE" \
        --filter-expression "instance_id = :instance_id" \
        --expression-attribute-values "{\":instance_id\": {\"S\": \"$INSTANCE_ID\"}}" \
        --query 'Items[0].job_id.S' \
        --output text \
        --region "$REGION" 2>/dev/null || echo "")
    
    if [[ -z "$JOB_ID" ]]; then
        # Last resort: find most recent launching job
        JOB_ID=$(aws dynamodb scan \
            --table-name "$DYNAMODB_TABLE" \
            --filter-expression "#status = :status" \
            --expression-attribute-names '{"#status": "status"}' \
            --expression-attribute-values "{\":status\": {\"S\": \"launching\"}}" \
            --query 'Items[0].job_id.S' \
            --output text \
            --region "$REGION" 2>/dev/null || echo "")
    fi
fi

if [[ -z "$JOB_ID" ]]; then
    log_msg "Error: Could not find job ID"
    exit 1
fi

log_msg "Job ID: $JOB_ID"
export JOB_ID

log_msg "=== Fast GPU Transcription Startup ===" "STATUS"

# Use environment variables passed from Lambda (no DynamoDB lookup needed)
if [[ -z "$S3_KEY" ]]; then
    log_msg "Error: S3_KEY environment variable not set"
    exit 1
fi

if [[ -z "$STANDARDIZED_FILENAME" ]]; then
    log_msg "Error: STANDARDIZED_FILENAME environment variable not set"
    exit 1
fi

log_msg "S3 Key: $S3_KEY"
log_msg "Standardized filename: $STANDARDIZED_FILENAME"

# Update status to processing
aws dynamodb update-item \
    --table-name "$DYNAMODB_TABLE" \
    --key "{\"job_id\": {\"S\": \"$JOB_ID\"}}" \
    --update-expression "SET #status = :status, updated_at = :updated_at" \
    --expression-attribute-names '{"#status": "status"}' \
    --expression-attribute-values "{\":status\": {\"S\": \"processing\"}, \":updated_at\": {\"S\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}}" \
    --region "$REGION"

log_msg "Status updated to processing" "STATUS"

# AMI verification - check critical components
log_msg "Verifying AMI setup..." "STATUS"

if [[ ! -f "/opt/transcribe/.setup_complete" ]]; then
    log_msg "ERROR: AMI setup incomplete - missing /opt/transcribe/.setup_complete" "ERROR"
    log_msg "AMI $AMI_ID was not properly built or setup failed" "ERROR"
    exit 1
fi

# Check critical paths and dependencies
MISSING_COMPONENTS=()

if [[ ! -f "/opt/transcribe/fast_transcribe.py" ]]; then
    MISSING_COMPONENTS+=("Python script: /opt/transcribe/fast_transcribe.py")
fi

if [[ ! -d "/opt/transcribe/venv" ]]; then
    MISSING_COMPONENTS+=("Virtual environment: /opt/transcribe/venv")
fi

if [[ ! -d "/opt/transcribe/models" ]]; then
    MISSING_COMPONENTS+=("Model cache: /opt/transcribe/models")
fi

if [[ ! -f "/opt/transcribe/gpu_state/model_gpu_state.pt" ]]; then
    MISSING_COMPONENTS+=("GPU state: /opt/transcribe/gpu_state/model_gpu_state.pt")
fi

if [[ ! -d "/opt/transcribe/gpu_state/processor" ]]; then
    MISSING_COMPONENTS+=("GPU processor: /opt/transcribe/gpu_state/processor")
fi

if [[ ${#MISSING_COMPONENTS[@]} -gt 0 ]]; then
    log_msg "ERROR: Missing AMI components:" "ERROR"
    for component in "${MISSING_COMPONENTS[@]}"; do
        log_msg "  - $component" "ERROR"
    done
    log_msg "AMI $AMI_ID appears to be incomplete" "ERROR"
    exit 1
fi

log_msg "AMI verified - all dependencies ready"

# Setup working directory
cd /opt/transcribe
log_msg "Current working directory: $(pwd)" "DEBUG"

# Verify S3 object exists before attempting download
log_msg "Verifying S3 object exists: s3://$S3_BUCKET/$S3_KEY" "DEBUG"
if ! aws s3api head-object --bucket "$S3_BUCKET" --key "$S3_KEY" --region "$REGION" >/dev/null 2>&1; then
    log_msg "ERROR: S3 object does not exist: s3://$S3_BUCKET/$S3_KEY" "ERROR"
    log_msg "This indicates a failure in the Lambda file processing step" "ERROR"
    exit 1
fi

# Download audio file with standardized name
log_msg "Downloading audio file... S3_BUCKET=$S3_BUCKET S3_KEY=$S3_KEY FILENAME=$STANDARDIZED_FILENAME" "DEBUG"
aws s3 cp "s3://$S3_BUCKET/$S3_KEY" "./$STANDARDIZED_FILENAME" --region "$REGION"

# Verify download
if [[ ! -f "./$STANDARDIZED_FILENAME" ]]; then
    log_msg "Error: Download failed"
    exit 1
fi

FILE_SIZE=$(stat -c%s "./$STANDARDIZED_FILENAME" 2>/dev/null || echo "0")
log_msg "Downloaded: $FILE_SIZE bytes"

# Activate virtual environment (pre-installed)
log_msg "Activating virtual environment..." "STATUS"

if [[ ! -f "/opt/transcribe/venv/bin/activate" ]]; then
    log_msg "ERROR: Virtual environment activation script missing" "ERROR"
    log_msg "Expected: /opt/transcribe/venv/bin/activate" "ERROR"
    exit 1
fi

if ! source /opt/transcribe/venv/bin/activate; then
    log_msg "ERROR: Failed to activate virtual environment" "ERROR"
    exit 1
fi

# Verify Python environment
if ! command -v python3 &> /dev/null; then
    log_msg "ERROR: Python3 not available after venv activation" "ERROR"
    exit 1
fi

log_msg "Virtual environment activated successfully"

# Start transcription immediately
log_msg "=== Starting Transcription ===" "STATUS"
START_TIME=$(date +%s)

# Verify transcription script exists
if [[ ! -f "/opt/transcribe/fast_transcribe.py" ]]; then
    log_msg "ERROR: Transcription script missing: /opt/transcribe/fast_transcribe.py" "ERROR"
    exit 1
fi

# Verify audio file exists before transcription
if [[ ! -f "$STANDARDIZED_FILENAME" ]]; then
    log_msg "ERROR: Audio file missing: $STANDARDIZED_FILENAME" "ERROR"
    exit 1
fi

log_msg "Starting transcription: python3 /opt/transcribe/fast_transcribe.py $STANDARDIZED_FILENAME"

# Run transcription with detailed error capture
if python3 /opt/transcribe/fast_transcribe.py "$STANDARDIZED_FILENAME" 2>&1; then
    log_msg "Transcription completed successfully!" "STATUS"
    TRANSCRIPTION_SUCCESS=true
else
    TRANSCRIPTION_EXIT_CODE=$?
    log_msg "ERROR: Transcription failed with exit code $TRANSCRIPTION_EXIT_CODE" "ERROR"
    log_msg "Check logs above for specific Python errors" "ERROR"
    TRANSCRIPTION_SUCCESS=false
fi

# Update final status
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

# Send webhook via Lambda for reliability
CALLBACK_URL=$(aws dynamodb get-item \
    --table-name "$DYNAMODB_TABLE" \
    --key "{\"job_id\": {\"S\": \"$JOB_ID\"}}" \
    --query 'Item.callback_url.S' \
    --output text \
    --region "$REGION" 2>/dev/null || echo "")

if [[ -n "$CALLBACK_URL" && "$CALLBACK_URL" != "None" ]]; then
    log_msg "Sending webhook via Lambda..."
    
    # Get transcription text
    TRANSCRIPTION_TEXT=""
    if ls transcription_*.txt 1> /dev/null 2>&1; then
        LATEST_FILE=$(ls -t transcription_*.txt | head -1)
        if [[ -n "$LATEST_FILE" ]]; then
            TRANSCRIPTION_TEXT=$(sed -n '/=== TRANSCRIPTION ===/,$p' "$LATEST_FILE" | tail -n +2 | tr '\n' ' ' | sed 's/"/\\"/g')
        fi
    fi
    
    if [[ -z "$TRANSCRIPTION_TEXT" ]]; then
        TRANSCRIPTION_TEXT="Transcription completed"
    fi
    
    TOTAL_TIME=$(($(date +%s) - START_TIME))
    
    # Prepare webhook payload
    WEBHOOK_PAYLOAD="{
        \"job_id\": \"$JOB_ID\",
        \"status\": \"$FINAL_STATUS\",
        \"transcript\": \"$TRANSCRIPTION_TEXT\",
        \"processing_time\": \"${TOTAL_TIME}s\",
        \"file_size\": \"$FILE_SIZE\"
    }"
    
    # Invoke Lambda for webhook delivery
    aws lambda invoke \
        --function-name WebhookDelivery \
        --payload "{
            \"job_id\": \"$JOB_ID\",
            \"callback_url\": \"$CALLBACK_URL\",
            \"payload\": $WEBHOOK_PAYLOAD
        }" \
        --region "$REGION" \
        /tmp/webhook_response.json >/dev/null 2>&1 || log_msg "Warning: Lambda webhook invocation failed"
    
    log_msg "Webhook sent via Lambda"
fi



# Show results
TOTAL_TIME=$(($(date +%s) - START_TIME))
log_msg "=== Complete ==="
log_msg "Total time: ${TOTAL_TIME}s"
log_msg "Job ID: $JOB_ID"

# Show transcription
LATEST_FILE=$(ls -t transcription_*.txt 2>/dev/null | head -1 || true)
if [[ -n "$LATEST_FILE" ]]; then
    log_msg "=== Transcription ==="
    sed -n '/=== TRANSCRIPTION ===/,$p' "$LATEST_FILE" | tail -n +2
fi

log_msg "Fast transcription completed!"

# Terminate instance
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null 2>&1 || true 