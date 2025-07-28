#!/bin/bash
# Verify and set Lambda environment variables

set -e

REGION="eu-north-1"
FUNCTION_NAME="TranscriptionProcessUpload"

log() {
    echo -e "\033[0;32m[$(date +'%Y-%m-%d %H:%M:%S')] $1\033[0m"
}

warn() {
    echo -e "\033[1;33m[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1\033[0m"
}

error() {
    echo -e "\033[0;31m[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1\033[0m"
    exit 1
}

# Check prerequisites
if ! command -v aws &> /dev/null; then
    error "AWS CLI not installed"
fi

if ! aws sts get-caller-identity &> /dev/null; then
    error "AWS credentials not configured"
fi

log "=== Verifying Lambda Environment Variables ==="

# Get current environment variables
log "Getting current environment variables..."
CURRENT_ENV=$(aws lambda get-function-configuration \
    --function-name $FUNCTION_NAME \
    --region $REGION \
    --query 'Environment.Variables' \
    --output json 2>/dev/null || echo "{}")

log "Current environment variables:"
echo "$CURRENT_ENV" | jq '.'

# Check and get missing variables
MISSING_VARS=()

# Check AMI_ID
AMI_ID=$(echo "$CURRENT_ENV" | jq -r '.AMI_ID // empty')
if [ -z "$AMI_ID" ]; then
    if [ -f "../run_transcription/ami_id.txt" ]; then
        AMI_ID=$(cat ../run_transcription/ami_id.txt)
        log "Found AMI ID: $AMI_ID"
    else
        warn "AMI_ID not found in environment or ami_id.txt"
        MISSING_VARS+=("AMI_ID")
    fi
fi

# Check SECURITY_GROUP_ID
SECURITY_GROUP_ID=$(echo "$CURRENT_ENV" | jq -r '.SECURITY_GROUP_ID // empty')
if [ -z "$SECURITY_GROUP_ID" ]; then
    SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=transcription-g4dn-sg" \
        --query 'SecurityGroups[0].GroupId' \
        --output text \
        --region $REGION 2>/dev/null || echo "")
    
    if [ -z "$SECURITY_GROUP_ID" ] || [ "$SECURITY_GROUP_ID" = "None" ]; then
        warn "SECURITY_GROUP_ID not found"
        MISSING_VARS+=("SECURITY_GROUP_ID")
    else
        log "Found Security Group ID: $SECURITY_GROUP_ID"
    fi
fi

# Check SUBNET_ID
SUBNET_ID=$(echo "$CURRENT_ENV" | jq -r '.SUBNET_ID // empty')
if [ -z "$SUBNET_ID" ]; then
    SUBNET_ID=$(aws ec2 describe-subnets \
        --filters "Name=availability-zone,Values=eu-north-1a" "Name=state,Values=available" \
        --query 'Subnets[0].SubnetId' \
        --output text \
        --region $REGION 2>/dev/null || echo "")
    
    if [ -z "$SUBNET_ID" ] || [ "$SUBNET_ID" = "None" ]; then
        warn "SUBNET_ID not found"
        MISSING_VARS+=("SUBNET_ID")
    else
        log "Found Subnet ID: $SUBNET_ID"
    fi
fi

# Check IAM_ROLE_NAME
IAM_ROLE_NAME=$(echo "$CURRENT_ENV" | jq -r '.IAM_ROLE_NAME // empty')
if [ -z "$IAM_ROLE_NAME" ]; then
    IAM_ROLE_NAME="EC2TranscriptionRole"
    log "Using default IAM Role Name: $IAM_ROLE_NAME"
fi

# Check INSTANCE_TYPE
INSTANCE_TYPE=$(echo "$CURRENT_ENV" | jq -r '.INSTANCE_TYPE // empty')
if [ -z "$INSTANCE_TYPE" ]; then
    INSTANCE_TYPE="g4dn.xlarge"
    log "Using default Instance Type: $INSTANCE_TYPE"
fi

# Check DYNAMODB_TABLE
DYNAMODB_TABLE=$(echo "$CURRENT_ENV" | jq -r '.DYNAMODB_TABLE // empty')
if [ -z "$DYNAMODB_TABLE" ]; then
    DYNAMODB_TABLE="transcription-jobs"
    log "Using default DynamoDB Table: $DYNAMODB_TABLE"
fi

# Update environment variables if we have all required values
if [ ${#MISSING_VARS[@]} -eq 0 ]; then
    log "All required environment variables found. Updating Lambda configuration..."
    
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --environment Variables="{
            DYNAMODB_TABLE=$DYNAMODB_TABLE,
            AMI_ID=$AMI_ID,
            INSTANCE_TYPE=$INSTANCE_TYPE,
            SECURITY_GROUP_ID=$SECURITY_GROUP_ID,
            SUBNET_ID=$SUBNET_ID,
            IAM_ROLE_NAME=$IAM_ROLE_NAME
        }" \
        --region $REGION
    
    log "✓ Lambda environment variables updated successfully"
else
    warn "Missing environment variables: ${MISSING_VARS[*]}"
    log ""
    log "Please set these variables manually in the AWS Lambda console:"
    for var in "${MISSING_VARS[@]}"; do
        case $var in
            "AMI_ID")
                log "  - AMI_ID: Run ./build_ami.sh to create AMI, then copy AMI ID from ami_id.txt"
                ;;
            "SECURITY_GROUP_ID")
                log "  - SECURITY_GROUP_ID: Run ./setup_infrastructure.sh to create security group"
                ;;
            "SUBNET_ID")
                log "  - SUBNET_ID: Use a subnet ID from eu-north-1a availability zone"
                ;;
        esac
    done
fi

log "=== Environment Variable Verification Complete ===" 