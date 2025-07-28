#!/bin/bash

# Setup CloudWatch Logs for Transcription Service
# ==============================================

set -euo pipefail

REGION="eu-north-1"
LOG_GROUP="/aws/ec2/transcription"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "=== Setting up CloudWatch Logs ==="

# Check if AWS CLI is configured
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    log "Error: AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi

# Create CloudWatch log group
log "Creating CloudWatch log group: $LOG_GROUP"
aws logs create-log-group \
    --log-group-name "$LOG_GROUP" \
    --region "$REGION" 2>/dev/null || log "Log group may already exist"

# Set retention policy (7 days)
log "Setting retention policy to 7 days"
aws logs put-retention-policy \
    --log-group-name "$LOG_GROUP" \
    --retention-in-days 7 \
    --region "$REGION"

log "CloudWatch Logs setup complete ✓"
log "Log group: $LOG_GROUP"
log "Retention: 7 days"
log "Region: $REGION" 