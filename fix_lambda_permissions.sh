#!/bin/bash

# Fix Lambda Role Permissions
# ==========================
# Add missing ec2:CreateTags permission to TranscriptionLambdaRole

set -euo pipefail

REGION="eu-north-1"
ROLE_NAME="TranscriptionLambdaRole"
POLICY_NAME="TranscriptionLambdaPolicy"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "=== Fixing Lambda Role Permissions ==="

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    log "Error: AWS CLI not found"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    log "Error: AWS credentials not configured"
    exit 1
fi

log "Updating Lambda role permissions..."

# Create updated policy document with all required permissions
cat > /tmp/lambda_policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::transcription-curevo",
                "arn:aws:s3:::transcription-curevo/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GeneratePresignedPost", "s3:GeneratePresignedUrl"
            ],
            "Resource": "arn:aws:s3:::transcription-curevo/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem",
                "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan",
                "dynamodb:BatchGetItem", "dynamodb:BatchWriteItem"
            ],
            "Resource": "arn:aws:dynamodb:*:*:table/transcription-jobs"
        },
        {
            "Effect": "Allow",
            "Action": [
                "dynamodb:CreateTable", "dynamodb:DescribeTable", "dynamodb:ListTables"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:RunInstances", "ec2:CreateTags", "ec2:DescribeInstances", 
                "ec2:TerminateInstances", "ec2:DescribeSecurityGroups", 
                "ec2:DescribeSubnets", "ec2:DescribeVpcs", "ec2:DescribeTags"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": "arn:aws:iam::*:role/EC2TranscriptionRole"
        }
    ]
}
EOF

# Update the role policy
aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "$POLICY_NAME" \
    --policy-document file:///tmp/lambda_policy.json \
    --region "$REGION"

log "✓ Lambda role permissions updated successfully"

# Clean up
rm -f /tmp/lambda_policy.json

log ""
log "=== Permission Fix Complete ==="
log "The Lambda function should now be able to create EC2 instances with tags."
log ""
log "Next steps:"
log "1. Re-upload your audio file to trigger the Lambda:"
log "   aws s3 cp audio20min.mp3 s3://transcription-curevo/transcription_upload/audio20min.mp3"
log ""
log "2. Check DynamoDB for the new job entry"
log "3. Check CloudWatch logs for the Lambda function"
log "4. Check EC2 console for the new instance" 