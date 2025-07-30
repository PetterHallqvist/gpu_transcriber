#!/bin/bash
# DynamoDB Setup - Creates table and configures permissions

set -euo pipefail

REGION="eu-north-1"
TABLE_NAME="transcription-jobs"
EC2_ROLE="EC2TranscriptionRole"
LAMBDA_ROLE="TranscriptionLambdaRole"

log() { echo -e "\033[0;32m[$(date +'%H:%M:%S')] $1\033[0m"; }
warn() { echo -e "\033[1;33m[$(date +'%H:%M:%S')] $1\033[0m"; }
error() { echo -e "\033[0;31m[$(date +'%H:%M:%S')] ERROR: $1\033[0m"; exit 1; }

# Check AWS CLI
command -v aws >/dev/null || error "AWS CLI not installed"
aws sts get-caller-identity >/dev/null || error "AWS credentials not configured"

log "Setting up DynamoDB table and permissions..."

# Create DynamoDB table
if ! aws dynamodb describe-table --table-name "$TABLE_NAME" --region "$REGION" >/dev/null 2>&1; then
    log "Creating DynamoDB table '$TABLE_NAME'..."
    aws dynamodb create-table \
        --table-name "$TABLE_NAME" \
        --attribute-definitions AttributeName=job_id,AttributeType=S \
        --key-schema AttributeName=job_id,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST \
        --region "$REGION"
    aws dynamodb wait table-exists --table-name "$TABLE_NAME" --region "$REGION"
    log "✓ Table created"
else
    log "✓ Table already exists"
fi

# EC2 Role Policy
aws iam put-role-policy --role-name "$EC2_ROLE" --policy-name TranscriptionEC2Policy --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                          "Resource": [
                "arn:aws:s3:::transcription-curevo/transcription_upload/*",
                "arn:aws:s3:::transcription-curevo/results/*",
                "arn:aws:s3:::transcription-curevo/transcription_results/*"
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
            "Action": ["ec2:DescribeTags"],
            "Resource": "*"
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
}' 2>/dev/null && log "✓ EC2 role updated" || warn "EC2 role not found"

# Lambda Role Policy
aws iam put-role-policy --role-name "$LAMBDA_ROLE" --policy-name TranscriptionLambdaPolicy --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
            "Resource": "arn:aws:logs:*:*:*"
        },
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
            "Resource": ["arn:aws:s3:::transcription-curevo", "arn:aws:s3:::transcription-curevo/*"]
        },
        {
            "Effect": "Allow",
            "Action": ["s3:GeneratePresignedPost", "s3:GeneratePresignedUrl"],
            "Resource": "arn:aws:s3:::transcription-curevo/*"
        },
        {
            "Effect": "Allow",
            "Action": ["ec2:RunInstances", "ec2:CreateTags", "ec2:DescribeInstances", "ec2:TerminateInstances"],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem",
                "dynamodb:DeleteItem", "dynamodb:Query", "dynamodb:Scan"
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
            "Action": ["iam:PassRole"],
            "Resource": "arn:aws:iam::*:role/EC2TranscriptionRole"
        }
    ]
}' 2>/dev/null && log "✓ Lambda role updated" || warn "Lambda role not found"

log "✓ DynamoDB setup complete" 