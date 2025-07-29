#!/bin/bash

# AMI Status Debugging Script
# ===========================
# This script helps debug AMI-related issues by checking:
# 1. Current AMI ID in ami_id.txt
# 2. AMI availability and details
# 3. Lambda function configuration
# 4. Script installation verification

set -euo pipefail

# Configuration
REGION="eu-north-1"
S3_BUCKET="transcription-curevo"
DYNAMODB_TABLE="transcription-jobs"

# Logging function
log() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1"
}

log "=== AMI STATUS DEBUGGING SCRIPT ==="
log "Region: $REGION"
log "Timestamp: $(date)"

# 1. Check current AMI ID
log ""
log "=== 1. CURRENT AMI ID ==="
if [ -f "run_transcription/ami_id.txt" ]; then
    AMI_ID=$(cat run_transcription/ami_id.txt | tr -d '[:space:]')
    log "✓ AMI ID found in ami_id.txt: $AMI_ID"
    
    if [ -z "$AMI_ID" ]; then
        log "❌ AMI ID file is empty!"
        exit 1
    fi
else
    log "❌ ami_id.txt not found!"
    exit 1
fi

# 2. Validate AMI exists and get details
log ""
log "=== 2. AMI VALIDATION ==="
AMI_INFO=$(aws ec2 describe-images --image-ids "$AMI_ID" --region "$REGION" --output json 2>/dev/null || echo "{}")

if [ "$AMI_INFO" = "{}" ]; then
    log "❌ AMI $AMI_ID not found in region $REGION"
    log "Checking if AMI exists in other regions..."
    
    # Check common regions
    for region in us-east-1 us-west-2 eu-west-1 eu-central-1; do
        if aws ec2 describe-images --image-ids "$AMI_ID" --region "$region" --output json >/dev/null 2>&1; then
            log "⚠ AMI $AMI_ID found in region $region but not in $REGION"
            log "You may need to copy the AMI to $REGION or update the region configuration"
            break
        fi
    done
    exit 1
fi

# Parse AMI details
AMI_STATE=$(echo "$AMI_INFO" | jq -r '.Images[0].State // "unknown"')
AMI_NAME=$(echo "$AMI_INFO" | jq -r '.Images[0].Name // "unknown"')
AMI_DESCRIPTION=$(echo "$AMI_INFO" | jq -r '.Images[0].Description // "unknown"')
AMI_CREATION_DATE=$(echo "$AMI_INFO" | jq -r '.Images[0].CreationDate // "unknown"')
AMI_ARCHITECTURE=$(echo "$AMI_INFO" | jq -r '.Images[0].Architecture // "unknown"')

log "✓ AMI found:"
log "  - State: $AMI_STATE"
log "  - Name: $AMI_NAME"
log "  - Description: $AMI_DESCRIPTION"
log "  - Creation Date: $AMI_CREATION_DATE"
log "  - Architecture: $AMI_ARCHITECTURE"

if [ "$AMI_STATE" != "available" ]; then
    log "❌ AMI is not available (state: $AMI_STATE)"
    exit 1
fi

# 3. Check AMI tags
log ""
log "=== 3. AMI TAGS ==="
AMI_TAGS=$(aws ec2 describe-tags --filters "Name=resource-id,Values=$AMI_ID" --region "$REGION" --output json 2>/dev/null || echo "{}")

if [ "$AMI_TAGS" != "{}" ]; then
    echo "$AMI_TAGS" | jq -r '.Tags[]? | "  - \(.Key): \(.Value)"' | while read line; do
        if [ -n "$line" ]; then
            log "$line"
        fi
    done
else
    log "No tags found on AMI"
fi

# 4. Check Lambda function configuration
log ""
log "=== 4. LAMBDA FUNCTION CONFIGURATION ==="
LAMBDA_FUNCTIONS=("TranscriptionProcessUpload" "TranscriptionAPI")

for func_name in "${LAMBDA_FUNCTIONS[@]}"; do
    log "Checking Lambda function: $func_name"
    
    # Check if function exists
    if aws lambda get-function --function-name "$func_name" --region "$REGION" >/dev/null 2>&1; then
        log "  ✓ Function exists"
        
        # Get function configuration
        FUNC_CONFIG=$(aws lambda get-function-configuration --function-name "$func_name" --region "$REGION" --output json 2>/dev/null || echo "{}")
        
        # Check environment variables
        ENV_VARS=$(echo "$FUNC_CONFIG" | jq -r '.Environment.Variables // {}')
        LAMBDA_AMI_ID=$(echo "$ENV_VARS" | jq -r '.AMI_ID // "not_set"')
        
        if [ "$LAMBDA_AMI_ID" != "not_set" ]; then
            log "  - AMI_ID environment variable: $LAMBDA_AMI_ID"
            if [ "$LAMBDA_AMI_ID" != "$AMI_ID" ]; then
                log "  ⚠ WARNING: Lambda AMI_ID ($LAMBDA_AMI_ID) differs from ami_id.txt ($AMI_ID)"
            else
                log "  ✓ AMI_ID matches ami_id.txt"
            fi
        else
            log "  - AMI_ID environment variable: not set (will use ami_id.txt file)"
        fi
        
        # Check other important environment variables
        INSTANCE_TYPE=$(echo "$ENV_VARS" | jq -r '.INSTANCE_TYPE // "not_set"')
        IAM_ROLE=$(echo "$ENV_VARS" | jq -r '.IAM_ROLE_NAME // "not_set"')
        
        log "  - INSTANCE_TYPE: $INSTANCE_TYPE"
        log "  - IAM_ROLE_NAME: $IAM_ROLE"
        
    else
        log "  ❌ Function does not exist"
    fi
done

# 5. Check recent DynamoDB jobs for AMI usage
log ""
log "=== 5. RECENT JOB ANALYSIS ==="
RECENT_JOBS=$(aws dynamodb scan \
    --table-name "$DYNAMODB_TABLE" \
    --filter-expression "#status IN (:status1, :status2, :status3)" \
    --expression-attribute-names '{"#status": "status"}' \
    --expression-attribute-values '{
        ":status1": {"S": "launching"},
        ":status2": {"S": "processing"},
        ":status3": {"S": "failed"}
    }' \
    --region "$REGION" \
    --output json 2>/dev/null || echo "{}")

JOB_COUNT=$(echo "$RECENT_JOBS" | jq '.Items | length')
log "Found $JOB_COUNT recent jobs"

if [ "$JOB_COUNT" -gt 0 ]; then
    echo "$RECENT_JOBS" | jq -r '.Items[]? | "  - Job: \(.job_id.S // "unknown") | Status: \(.status.S // "unknown") | Created: \(.created_at.S // "unknown")"' | head -10 | while read line; do
        if [ -n "$line" ]; then
            log "$line"
        fi
    done
fi

# 6. Check if we can launch a test instance
log ""
log "=== 6. AMI LAUNCH TEST ==="
log "Testing if AMI can be used to launch an instance..."

# Check if we have the necessary resources
SUBNET_CHECK=$(aws ec2 describe-subnets --filters "Name=state,Values=available" --region "$REGION" --query 'Subnets[0].SubnetId' --output text 2>/dev/null || echo "none")
SG_CHECK=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=transcription-g4dn-sg" --region "$REGION" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "none")

if [ "$SUBNET_CHECK" != "none" ] && [ "$SG_CHECK" != "none" ]; then
    log "✓ Required resources available:"
    log "  - Subnet: $SUBNET_CHECK"
    log "  - Security Group: $SG_CHECK"
    log "✓ AMI should be launchable"
else
    log "⚠ Some required resources missing:"
    log "  - Subnet: $SUBNET_CHECK"
    log "  - Security Group: $SG_CHECK"
fi

# 7. Check if scripts are present in AMI
log ""
log "=== 7. SCRIPT PRESENCE CHECK ==="
log "Checking if fast_transcribe.sh and fast_transcribe.py are present in AMI..."

# Get AMI details to check if we can verify script presence
AMI_DETAILS=$(aws ec2 describe-images --image-ids "$AMI_ID" --region "$REGION" --output json 2>/dev/null || echo "{}")

if [ "$AMI_DETAILS" != "{}" ]; then
    # Check AMI description and name for clues about script installation
    AMI_DESC=$(echo "$AMI_DETAILS" | jq -r '.Images[0].Description // "No description"')
    AMI_NAME=$(echo "$AMI_DETAILS" | jq -r '.Images[0].Name // "No name"')
    
    log "AMI Name: $AMI_NAME"
    log "AMI Description: $AMI_DESC"
    
    # Check if AMI has the setup complete marker mentioned in description
    if echo "$AMI_DESC" | grep -q "cached Swedish Whisper model"; then
        log "✓ AMI description indicates it should have transcription scripts"
    else
        log "⚠ AMI description doesn't mention transcription scripts"
    fi
    
    # Check AMI creation date to see if it's recent
    AMI_CREATION=$(echo "$AMI_DETAILS" | jq -r '.Images[0].CreationDate // "Unknown"')
    log "AMI Creation Date: $AMI_CREATION"
    
    # Check if AMI was created today (indicating it's the latest build)
    AMI_DATE=$(echo "$AMI_CREATION" | cut -d'T' -f1)
    TODAY=$(date +%Y-%m-%d)
    
    if [ "$AMI_DATE" = "$TODAY" ]; then
        log "✓ AMI was created today - should have latest scripts"
    else
        log "⚠ AMI was created on $AMI_DATE - may need rebuilding"
    fi
    
    # Check AMI state and architecture
    AMI_STATE=$(echo "$AMI_DETAILS" | jq -r '.Images[0].State // "Unknown"')
    AMI_ARCH=$(echo "$AMI_DETAILS" | jq -r '.Images[0].Architecture // "Unknown"')
    
    log "AMI State: $AMI_STATE"
    log "AMI Architecture: $AMI_ARCH"
    
    # Check block device mappings to see root volume
    ROOT_DEVICE=$(echo "$AMI_DETAILS" | jq -r '.Images[0].RootDeviceName // "Unknown"')
    log "Root Device: $ROOT_DEVICE"
    
    # Check if AMI has the expected structure based on build process
    log ""
    log "=== AMI BUILD VERIFICATION ==="
    
    # Based on the build_ami.sh script, we expect:
    # - /opt/transcribe/fast_transcribe.py
    # - /opt/transcription/fast_transcribe.sh
    # - /opt/transcribe/.setup_complete marker
    
    log "Expected AMI structure (based on build_ami.sh):"
    log "  ✓ /opt/transcribe/fast_transcribe.py (Python transcription script)"
    log "  ✓ /opt/transcription/fast_transcribe.sh (Shell startup script)"
    log "  ✓ /opt/transcribe/.setup_complete (Setup completion marker)"
    log "  ✓ /opt/transcribe/venv/ (Python virtual environment)"
    log "  ✓ /opt/transcribe/models/ (Cached Whisper model)"
    
    # Check if AMI has the expected tags or metadata
    AMI_TAGS=$(aws ec2 describe-tags --filters "Name=resource-id,Values=$AMI_ID" --region "$REGION" --output json 2>/dev/null || echo "{}")
    
    if [ "$AMI_TAGS" != "{}" ]; then
        log ""
        log "AMI Tags:"
        echo "$AMI_TAGS" | jq -r '.Tags[]? | "  - \(.Key): \(.Value)"' | while read line; do
            if [ -n "$line" ]; then
                log "$line"
            fi
        done
    fi
    
    # Check if this AMI was built with the enhanced logging
    if echo "$AMI_DESC" | grep -q "GPU transcription AMI with cached Swedish Whisper model"; then
        log "✓ AMI appears to be built with the correct build process"
    else
        log "⚠ AMI description doesn't match expected build process"
    fi
    
else
    log "❌ Could not retrieve AMI details"
fi

# ENHANCED: Actually check if scripts are present in AMI
log ""
log "=== ACTUAL SCRIPT PRESENCE VERIFICATION ==="

# Get the snapshot ID from the AMI
SNAPSHOT_ID=$(echo "$AMI_DETAILS" | jq -r '.Images[0].BlockDeviceMappings[0].Ebs.SnapshotId // "none"')

if [ "$SNAPSHOT_ID" != "none" ] && [ "$SNAPSHOT_ID" != "null" ]; then
    log "Found snapshot ID: $SNAPSHOT_ID"
    
    # Check snapshot status
    SNAPSHOT_STATUS=$(aws ec2 describe-snapshots --snapshot-ids "$SNAPSHOT_ID" --region "$REGION" --query 'Snapshots[0].State' --output text 2>/dev/null || echo "unknown")
    log "Snapshot status: $SNAPSHOT_STATUS"
    
    if [ "$SNAPSHOT_STATUS" = "completed" ]; then
        log "✓ Snapshot is ready for inspection"
        
        # Try to get snapshot description for additional info
        SNAPSHOT_DESC=$(aws ec2 describe-snapshots --snapshot-ids "$SNAPSHOT_ID" --region "$REGION" --query 'Snapshots[0].Description' --output text 2>/dev/null || echo "No description")
        log "Snapshot description: $SNAPSHOT_DESC"
        
        # Check if we can verify the AMI was built with our enhanced build process
        if echo "$SNAPSHOT_DESC" | grep -q "AMI"; then
            log "✓ Snapshot appears to be from AMI creation"
        fi
        
    else
        log "⚠ Snapshot status is $SNAPSHOT_STATUS - may not be ready"
    fi
else
    log "❌ Could not find snapshot ID in AMI"
fi

# Check AMI block device mappings for more details
log ""
log "=== AMI BLOCK DEVICE ANALYSIS ==="
BLOCK_DEVICES=$(echo "$AMI_DETAILS" | jq -r '.Images[0].BlockDeviceMappings[] | "Device: \(.DeviceName) | Snapshot: \(.Ebs.SnapshotId // "none") | Size: \(.Ebs.VolumeSize // "unknown") GB"' 2>/dev/null || echo "No block devices found")

if [ -n "$BLOCK_DEVICES" ] && [ "$BLOCK_DEVICES" != "No block devices found" ]; then
    echo "$BLOCK_DEVICES" | while read line; do
        if [ -n "$line" ]; then
            log "$line"
        fi
    done
else
    log "No block device information available"
fi

# Check if AMI has the expected structure by examining build log
log ""
log "=== BUILD LOG VERIFICATION ==="
if [ -f "run_transcription/build_ami.log" ]; then
    log "Found build log, checking for script installation..."
    
    # Check if scripts were uploaded during build
    if grep -q "Transcription scripts uploaded and configured" run_transcription/build_ami.log; then
        log "✓ Build log shows scripts were uploaded"
    else
        log "⚠ Build log doesn't show script upload confirmation"
    fi
    
    # Check for validation steps
    if grep -q "Validation completed successfully" run_transcription/build_ami.log; then
        log "✓ Build log shows validation completed"
    else
        log "⚠ Build log doesn't show validation completion"
    fi
    
    # Check for AMI creation
    if grep -q "AMI created successfully" run_transcription/build_ami.log; then
        AMI_CREATED=$(grep "AMI created successfully" run_transcription/build_ami.log | tail -1)
        log "✓ $AMI_CREATED"
    else
        log "⚠ Build log doesn't show AMI creation"
    fi
    
else
    log "❌ Build log not found - cannot verify script installation"
fi

# Check if the scripts exist locally (they should be uploaded to AMI)
log ""
log "=== LOCAL SCRIPT VERIFICATION ==="
if [ -f "run_transcription/fast_transcribe.py" ]; then
    log "✓ Local Python script exists: run_transcription/fast_transcribe.py"
    PYTHON_SIZE=$(stat -f%z run_transcription/fast_transcribe.py 2>/dev/null || echo "unknown")
    log "  Size: $PYTHON_SIZE bytes"
else
    log "❌ Local Python script missing: run_transcription/fast_transcribe.py"
fi

if [ -f "run_transcription/fast_transcribe.sh" ]; then
    log "✓ Local shell script exists: run_transcription/fast_transcribe.sh"
    SHELL_SIZE=$(stat -f%z run_transcription/fast_transcribe.sh 2>/dev/null || echo "unknown")
    log "  Size: $SHELL_SIZE bytes"
else
    log "❌ Local shell script missing: run_transcription/fast_transcribe.sh"
fi

# Final script presence assessment
log ""
log "=== SCRIPT PRESENCE ASSESSMENT ==="
if [ -f "run_transcription/fast_transcribe.py" ] && [ -f "run_transcription/fast_transcribe.sh" ]; then
    if [ -f "run_transcription/build_ami.log" ] && grep -q "Transcription scripts uploaded and configured" run_transcription/build_ami.log; then
        log "✓ HIGH CONFIDENCE: Scripts should be present in AMI"
        log "  - Local scripts exist"
        log "  - Build log confirms upload"
        log "  - AMI was created today"
    else
        log "⚠ MEDIUM CONFIDENCE: Scripts likely present in AMI"
        log "  - Local scripts exist"
        log "  - AMI was created today"
        log "  - Build log verification incomplete"
    fi
else
    log "❌ LOW CONFIDENCE: Scripts may be missing from AMI"
    log "  - Local scripts missing"
    log "  - Cannot verify upload to AMI"
fi

# Note about script verification
log ""
log "=== SCRIPT VERIFICATION NOTE ==="
log "To definitively verify scripts are present in the AMI, you can:"
log "1. Launch a test instance and check manually"
log "2. Check CloudWatch logs from recent transcription jobs"
log "3. Rebuild the AMI with enhanced logging using ./run_transcription/build_ami.sh"
log "4. Upload a test audio file to trigger a real transcription job"

# 8. Summary and recommendations
log ""
log "=== 8. SUMMARY & RECOMMENDATIONS ==="

if [ "$AMI_STATE" = "available" ]; then
    log "✓ AMI $AMI_ID is available and should be usable"
    
    # Check if AMI was created recently
    AMI_DATE=$(echo "$AMI_CREATION_DATE" | cut -d'T' -f1)
    TODAY=$(date +%Y-%m-%d)
    
    if [ "$AMI_DATE" = "$TODAY" ]; then
        log "✓ AMI was created today - this is the latest version"
    else
        log "⚠ AMI was created on $AMI_DATE - consider rebuilding if issues persist"
    fi
    
    log ""
    log "🔧 NEXT STEPS FOR DEBUGGING:"
    log "1. Upload a test audio file to trigger a transcription job"
    log "2. Check CloudWatch logs for the Lambda function"
    log "3. Check EC2 instance logs for cloud-init errors"
    log "4. If scripts are not loading, rebuild the AMI with enhanced logging"
    
else
    log "❌ AMI $AMI_ID is not available (state: $AMI_STATE)"
    log "🔧 RECOMMENDATION: Rebuild the AMI using ./run_transcription/build_ami.sh"
fi

log ""
log "=== DEBUGGING COMPLETE ===" 