#!/bin/bash
# Deploy Lambda functions for transcription service

set -e

REGION="eu-north-1"
S3_BUCKET="transcription-curevo"
DYNAMODB_TABLE="transcription-jobs"
LAMBDA_ROLE_NAME="TranscriptionLambdaRole"
EC2_ROLE_NAME="EC2TranscriptionRole"
AMI_ID="ami-053c961804f4cb227"  # GPU-enabled AMI for transcription processing

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

if ! command -v jq &> /dev/null; then
    warn "jq not installed. Installing with brew..."
    if command -v brew &> /dev/null; then
        brew install jq
    else
        error "jq not installed and brew not available. Please install jq manually."
    fi
fi

log "=== Starting Lambda Deployment ==="

# Create DynamoDB table
log "Creating DynamoDB table..."
aws dynamodb create-table \
    --table-name $DYNAMODB_TABLE \
    --attribute-definitions AttributeName=job_id,AttributeType=S \
    --key-schema AttributeName=job_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region $REGION || warn "DynamoDB table may already exist"

aws dynamodb wait table-exists --table-name $DYNAMODB_TABLE --region $REGION

# Create EC2 instance role
log "Creating EC2 instance role..."
aws iam create-role \
    --role-name $EC2_ROLE_NAME \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    || warn "EC2 role may already exist"

aws iam put-role-policy \
    --role-name $EC2_ROLE_NAME \
    --policy-name TranscriptionEC2Policy \
    --policy-document file://../ec2_instance_role_policy.json

aws iam create-instance-profile --instance-profile-name $EC2_ROLE_NAME || warn "Instance profile may already exist"
aws iam add-role-to-instance-profile --instance-profile-name $EC2_ROLE_NAME --role-name $EC2_ROLE_NAME || warn "Role may already be attached"

# Create Lambda execution role
log "Creating Lambda execution role..."
aws iam create-role \
    --role-name $LAMBDA_ROLE_NAME \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    || warn "Lambda role may already exist"

aws iam put-role-policy \
    --role-name $LAMBDA_ROLE_NAME \
    --policy-name TranscriptionLambdaPolicy \
    --policy-document file://lambda_execution_role_policy.json

aws iam attach-role-policy \
    --role-name $LAMBDA_ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Get role ARNs
LAMBDA_ROLE_ARN=$(aws iam get-role --role-name $LAMBDA_ROLE_NAME --query 'Role.Arn' --output text)
EC2_ROLE_ARN=$(aws iam get-role --role-name $EC2_ROLE_NAME --query 'Role.Arn' --output text)

log "Lambda Role ARN: $LAMBDA_ROLE_ARN"
log "EC2 Role ARN: $EC2_ROLE_ARN"

# Create deployment packages
create_package() {
    local function_name=$1
    local source_file=$2
    
    log "Creating package for $function_name..."
    mkdir -p temp_$function_name
    cp $source_file temp_$function_name/
    cd temp_$function_name
    pip3 install boto3 -t . --quiet
    zip -r ../${function_name}.zip . --quiet
    cd ..
    rm -rf temp_$function_name
}

create_package "TranscriptionProcessUpload" "lambda_process_upload.py"
create_package "TranscriptionAPI" "lambda_api.py"

# Deploy Lambda functions
deploy_function() {
    local function_name=$1
    local zip_file=$2
    local handler=$3
    local timeout=${4:-300}
    local memory=${5:-512}
    
    log "Deploying $function_name..."
    
    if aws lambda get-function --function-name $function_name --region $REGION &> /dev/null; then
        log "Updating existing function: $function_name"
        aws lambda update-function-code --function-name $function_name --zip-file fileb://$zip_file --region $REGION
        
        # Check if configuration update is needed
        local current_config=$(aws lambda get-function-configuration --function-name $function_name --region $REGION 2>/dev/null || echo "{}")
        local current_timeout=$(echo "$current_config" | jq -r '.Timeout // "0"')
        local current_memory=$(echo "$current_config" | jq -r '.MemorySize // "0"')
        
        if [ "$current_timeout" != "$timeout" ] || [ "$current_memory" != "$memory" ]; then
            log "Updating function configuration (timeout: $current_timeout->$timeout, memory: $current_memory->$memory)"
            aws lambda update-function-configuration --function-name $function_name --timeout $timeout --memory-size $memory --region $REGION
        else
            log "Function configuration unchanged, skipping update"
        fi
    else
        log "Creating new function: $function_name"
        aws lambda create-function \
            --function-name $function_name \
            --runtime python3.9 \
            --role $LAMBDA_ROLE_ARN \
            --handler $handler \
            --zip-file fileb://$zip_file \
            --timeout $timeout \
            --memory-size $memory \
            --region $REGION
    fi
}

deploy_function "TranscriptionProcessUpload" "TranscriptionProcessUpload.zip" "lambda_process_upload.lambda_handler" 300 1024
deploy_function "TranscriptionAPI" "TranscriptionAPI.zip" "lambda_api.lambda_handler" 60 512

# Set environment variables
log "Setting environment variables..."

# AMI ID is predefined at the top of this script
log "Using hardcoded AMI ID: $AMI_ID"

# Get Security Group ID
log "Getting Security Group ID..."
SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=transcription-g4dn-sg" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region $REGION 2>/dev/null || echo "")

if [ -z "$SECURITY_GROUP_ID" ] || [ "$SECURITY_GROUP_ID" = "None" ]; then
    log "Warning: Security group 'transcription-g4dn-sg' not found. You'll need to set SECURITY_GROUP_ID manually."
    SECURITY_GROUP_ID=""
else
    log "Found Security Group ID: $SECURITY_GROUP_ID"
fi

# Get Subnet ID (use first available subnet in eu-north-1a)
log "Getting Subnet ID..."
SUBNET_ID=$(aws ec2 describe-subnets \
    --filters "Name=availability-zone,Values=eu-north-1a" "Name=state,Values=available" \
    --query 'Subnets[0].SubnetId' \
    --output text \
    --region $REGION 2>/dev/null || echo "")

if [ -z "$SUBNET_ID" ] || [ "$SUBNET_ID" = "None" ]; then
    log "Warning: No available subnet found in eu-north-1a. You'll need to set SUBNET_ID manually."
    SUBNET_ID=""
else
    log "Found Subnet ID: $SUBNET_ID"
fi

# Get IAM Role Name (already defined above)
IAM_ROLE_NAME="EC2TranscriptionRole"
log "Using IAM Role Name: $IAM_ROLE_NAME"

# Helper function to check if environment variables need updating
update_env_vars_if_needed() {
    local function_name=$1
    local new_env_vars=$2
    
    log "Checking environment variables for $function_name..."
    
    # Get current environment variables
    local current_config=$(aws lambda get-function-configuration --function-name $function_name --region $REGION 2>/dev/null || echo "{}")
    local current_env_vars=$(echo "$current_config" | jq -r '.Environment.Variables // "{}"' 2>/dev/null || echo "{}")
    
    # Compare current vs new environment variables
    if [ "$current_env_vars" = "$new_env_vars" ]; then
        log "Environment variables unchanged for $function_name, skipping update"
    else
        log "Updating environment variables for $function_name"
        aws lambda update-function-configuration \
            --function-name $function_name \
            --environment Variables="$new_env_vars" \
            --region $REGION
    fi
}

# Update TranscriptionProcessUpload Lambda environment variables
PROCESS_UPLOAD_ENV="{DYNAMODB_TABLE=$DYNAMODB_TABLE,AMI_ID=$AMI_ID,INSTANCE_TYPE=g4dn.xlarge,SECURITY_GROUP_ID=$SECURITY_GROUP_ID,SUBNET_ID=$SUBNET_ID,IAM_ROLE_NAME=$IAM_ROLE_NAME}"
update_env_vars_if_needed "TranscriptionProcessUpload" "$PROCESS_UPLOAD_ENV"

# Update TranscriptionAPI Lambda environment variables
API_ENV="{DYNAMODB_TABLE=$DYNAMODB_TABLE}"
update_env_vars_if_needed "TranscriptionAPI" "$API_ENV"

# Setup CloudWatch Logs
log "Setting up CloudWatch Logs..."
./setup_cloudwatch_logs.sh

# Setup S3
log "Setting up S3..."
aws s3 mb s3://$S3_BUCKET --region $REGION || warn "S3 bucket may already exist"

# Enable S3 bucket versioning (recommended for production)
log "Enabling S3 bucket versioning..."
aws s3api put-bucket-versioning \
    --bucket $S3_BUCKET \
    --versioning-configuration Status=Enabled \
    --region $REGION || warn "Bucket versioning may already be enabled"

# Create S3 bucket policy to allow Lambda access
log "Creating S3 bucket policy..."
BUCKET_POLICY='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowLambdaAccess",
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::transcription-curevo",
                "arn:aws:s3:::transcription-curevo/*"
            ]
        },
        {
            "Sid": "AllowEC2Access",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::849681699488:role/EC2TranscriptionRole"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::transcription-curevo/transcription_upload/*",
                "arn:aws:s3:::transcription-curevo/results/*"
            ]
        }
    ]
}'

# Check if bucket policy needs updating
log "Checking S3 bucket policy..."
current_policy=$(aws s3api get-bucket-policy --bucket $S3_BUCKET --region $REGION 2>/dev/null | jq -r '.Policy // "{}"' 2>/dev/null || echo "{}")
new_policy_compact=$(echo "$BUCKET_POLICY" | jq -c '.' 2>/dev/null || echo "$BUCKET_POLICY")

if [ "$current_policy" = "$new_policy_compact" ]; then
    log "S3 bucket policy unchanged, skipping update"
else
    log "Updating S3 bucket policy..."
    aws s3api put-bucket-policy \
        --bucket $S3_BUCKET \
        --policy "$BUCKET_POLICY" \
        --region $REGION
fi

# Scripts are now pre-installed in AMI - no S3 upload needed
log "Scripts will be pre-installed in AMI - skipping S3 upload"

# Setup S3 event notification
log "Setting up S3 event notification..."
PROCESS_LAMBDA_ARN=$(aws lambda get-function --function-name TranscriptionProcessUpload --region $REGION --query 'Configuration.FunctionArn' --output text)

# Check if notification configuration needs updating
log "Checking S3 notification configuration..."
current_notification=$(aws s3api get-bucket-notification-configuration --bucket $S3_BUCKET --region $REGION 2>/dev/null | jq -c '.LambdaFunctionConfigurations // []' 2>/dev/null || echo "[]")
new_notification_config="[{\"Id\":\"TranscriptionUploadTrigger\",\"LambdaFunctionArn\":\"$PROCESS_LAMBDA_ARN\",\"Events\":[\"s3:ObjectCreated:*\"],\"Filter\":{\"Key\":{\"FilterRules\":[{\"Name\":\"prefix\",\"Value\":\"transcription_upload/\"}]}}}]"

if [ "$current_notification" = "$new_notification_config" ]; then
    log "S3 notification configuration unchanged, skipping update"
else
    log "Updating S3 notification configuration..."
    aws s3api put-bucket-notification-configuration \
        --bucket $S3_BUCKET \
        --notification-configuration "{
            \"LambdaFunctionConfigurations\": [
                {
                    \"Id\": \"TranscriptionUploadTrigger\",
                    \"LambdaFunctionArn\": \"$PROCESS_LAMBDA_ARN\",
                    \"Events\": [\"s3:ObjectCreated:*\"],
                    \"Filter\": {
                        \"Key\": {
                            \"FilterRules\": [
                                {
                                    \"Name\": \"prefix\",
                                    \"Value\": \"transcription_upload/\"
                                }
                            ]
                        }
                    }
                }
            ]
        }" --region $REGION
fi

# Add Lambda permission for S3
log "Checking Lambda permission for S3..."
if aws lambda get-policy --function-name TranscriptionProcessUpload --region $REGION 2>/dev/null | jq -r '.Policy' 2>/dev/null | jq -e '.Statement[] | select(.Sid == "S3InvokePermission")' >/dev/null 2>&1; then
    log "Lambda permission for S3 already exists, skipping"
else
    log "Adding Lambda permission for S3..."
    aws lambda add-permission \
        --function-name TranscriptionProcessUpload \
        --statement-id S3InvokePermission \
        --action lambda:InvokeFunction \
        --principal s3.amazonaws.com \
        --source-arn arn:aws:s3:::$S3_BUCKET \
        --region $REGION
fi

# Clean up
rm -f *.zip

log "=== Lambda Deployment Complete ==="
log ""
log "Environment variables have been automatically configured:"
log "✓ AMI_ID: $AMI_ID (hardcoded)"
log "✓ INSTANCE_TYPE: g4dn.xlarge"
log "✓ SECURITY_GROUP_ID: $SECURITY_GROUP_ID"
log "✓ SUBNET_ID: $SUBNET_ID"
log "✓ IAM_ROLE_NAME: $IAM_ROLE_NAME"
log ""
log "Next steps:"
log "1. If any environment variables show as empty above, run:"
log "   ./verify_lambda_env.sh"
log "2. Run API Gateway setup:"
log "   chmod +x ../setup_api_gateway.sh && ../setup_api_gateway.sh"
log ""
log "3. Test by uploading an MP3 file to s3://$S3_BUCKET/transcription_upload/" 