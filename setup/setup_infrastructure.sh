#!/bin/bash

# AWS Infrastructure Setup for GPU Transcription
# =============================================
# One-time setup script for security groups, VPC, and IAM roles

set -euo pipefail

# Configuration
REGION="eu-north-1"
SECURITY_GROUP="transcription-g4dn-sg"
EC2_ROLE_NAME="EC2TranscriptionRole"
KEY_NAME="transcription-ec2"

# Logging function with timestamps
log_msg() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log_msg "=== AWS Infrastructure Setup ==="
log_msg "Region: $REGION"

# Function to check if AWS CLI is configured
check_aws_config() {
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log_msg "Error: AWS CLI not configured. Run 'aws configure' first."
        exit 1
    fi
    log_msg "AWS CLI configured ✓"
}

# Function to create or verify EC2 key pair
setup_key_pair() {
    log_msg "Checking EC2 key pair: $KEY_NAME"
    
    if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" >/dev/null 2>&1; then
        log_msg "Key pair $KEY_NAME already exists"
        if [[ ! -f "${KEY_NAME}.pem" ]]; then
            log_msg "Warning: Key file ${KEY_NAME}.pem not found locally"
            log_msg "You may need to download the private key from AWS Console"
        else
            log_msg "Key file ${KEY_NAME}.pem found locally ✓"
        fi
    else
        log_msg "Creating key pair: $KEY_NAME"
        aws ec2 create-key-pair \
            --key-name "$KEY_NAME" \
            --query 'KeyMaterial' \
            --output text \
            --region "$REGION" > "${KEY_NAME}.pem"
        
        chmod 400 "${KEY_NAME}.pem"
        log_msg "Key pair created and saved to ${KEY_NAME}.pem ✓"
    fi
}

# Function to create security group
create_security_group() {
    log_msg "Checking security group: $SECURITY_GROUP"
    
    # Check if security group exists
    if aws ec2 describe-security-groups --group-names "$SECURITY_GROUP" --region "$REGION" >/dev/null 2>&1; then
        log_msg "Security group $SECURITY_GROUP already exists ✓"
        return 0
    fi
    
    # Get default VPC
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text --region "$REGION")
    if [[ "$VPC_ID" == "None" || -z "$VPC_ID" ]]; then
        log_msg "Error: No default VPC found"
        exit 1
    fi
    
    log_msg "Using VPC: $VPC_ID"
    
    # Create security group
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP" \
        --description "Security group for GPU transcription instances" \
        --vpc-id "$VPC_ID" \
        --region "$REGION" \
        --query 'GroupId' \
        --output text)
    
    log_msg "Created security group: $SECURITY_GROUP_ID"
    
    # Add outbound rule for internet access
    aws ec2 authorize-security-group-egress \
        --group-id "$SECURITY_GROUP_ID" \
        --protocol -1 \
        --port -1 \
        --cidr 0.0.0.0/0 \
        --region "$REGION"
    
    log_msg "Added outbound internet access rule ✓"
    
    # Add inbound SSH rule
    aws ec2 authorize-security-group-ingress \
        --group-id "$SECURITY_GROUP_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region "$REGION"
    
    log_msg "Added inbound SSH rule ✓"
}

# Function to create or verify EC2 instance role
setup_ec2_role() {
    log_msg "Checking EC2 instance role: $EC2_ROLE_NAME"
    
    # Check if role exists
    if aws iam get-role --role-name "$EC2_ROLE_NAME" >/dev/null 2>&1; then
        log_msg "EC2 role $EC2_ROLE_NAME already exists ✓"
    else
        log_msg "Creating EC2 instance role..."
        aws iam create-role \
            --role-name "$EC2_ROLE_NAME" \
            --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
        
        aws iam put-role-policy \
            --role-name "$EC2_ROLE_NAME" \
            --policy-name TranscriptionEC2Policy \
            --policy-document '{
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                        "Resource": [
                            "arn:aws:s3:::transcription-curevo/transcription_upload/*",
                            "arn:aws:s3:::transcription-curevo/results/*"
                        ]
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem",
                            "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan",
                            "dynamodb:BatchGetItem", "dynamodb:BatchWriteItem"
                        ],
                        "Resource": "arn:aws:dynamodb:*:*:table/transcription-jobs"
                    },
                    {
                        "Effect": "Allow",
                        "Action": ["dynamodb:CreateTable", "dynamodb:DescribeTable", "dynamodb:ListTables"],
                        "Resource": "*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": ["ec2:TerminateInstances"],
                        "Resource": "*",
                        "Condition": {"StringEquals": {"ec2:ResourceTag/AutoTerminate": "true"}}
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                            "logs:DescribeLogGroups", "logs:DescribeLogStreams"
                        ],
                        "Resource": [
                            "arn:aws:logs:*:*:log-group:/aws/ec2/transcription",
                            "arn:aws:logs:*:*:log-group:/aws/ec2/transcription:log-stream:*"
                        ]
                    }
                ]
            }'
        
        log_msg "EC2 role created ✓"
    fi
    
    # Check if instance profile exists
    if aws iam get-instance-profile --instance-profile-name "$EC2_ROLE_NAME" >/dev/null 2>&1; then
        log_msg "Instance profile $EC2_ROLE_NAME already exists ✓"
    else
        log_msg "Creating instance profile..."
        aws iam create-instance-profile --instance-profile-name "$EC2_ROLE_NAME"
        aws iam add-role-to-instance-profile --instance-profile-name "$EC2_ROLE_NAME" --role-name "$EC2_ROLE_NAME"
        log_msg "Instance profile created ✓"
    fi
}

# Function to verify S3 bucket exists
verify_s3_bucket() {
    BUCKET_NAME="transcription-curevo"
    log_msg "Checking S3 bucket: $BUCKET_NAME"
    
    if aws s3 ls "s3://$BUCKET_NAME" >/dev/null 2>&1; then
        log_msg "S3 bucket $BUCKET_NAME exists ✓"
    else
        log_msg "Creating S3 bucket: $BUCKET_NAME"
        aws s3 mb "s3://$BUCKET_NAME" --region "$REGION"
        log_msg "S3 bucket created ✓"
    fi
}



# Main execution
main() {
    check_aws_config
    setup_key_pair
    create_security_group
    setup_ec2_role
    verify_s3_bucket
    
    log_msg ""
    log_msg "=== Infrastructure Setup Complete ==="
    log_msg "Security Group: $SECURITY_GROUP"
    log_msg "EC2 Role: $EC2_ROLE_NAME"
    log_msg "Key Pair: $KEY_NAME"
    log_msg "Region: $REGION"
    log_msg ""
    log_msg "Next steps:"
    log_msg "1. Run setup_dynamodb.sh for DynamoDB setup"
    log_msg "2. Run deploy_lambda_functions.sh for Lambda setup"
    log_msg "3. Run setup_api_gateway.sh for API Gateway setup"
    log_msg "4. Run fast_transcribe.sh to start transcription"
}

main "$@" 