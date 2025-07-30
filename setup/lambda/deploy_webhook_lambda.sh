#!/bin/bash

# Deploy Webhook Delivery Lambda
# =============================

set -euo pipefail

REGION="eu-north-1"
FUNCTION_NAME="WebhookDelivery"
RUNTIME="python3.9"
HANDLER="lambda_webhook_delivery.lambda_handler"
TIMEOUT=60
MEMORY_SIZE=128

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "=== Deploying Webhook Delivery Lambda ==="

# Create deployment package
log "Creating deployment package..."
cd "$(dirname "$0")"

# Install requests library
pip3 install requests -t temp_package/ --quiet

# Copy Lambda function
cp lambda_webhook_delivery.py temp_package/

# Create ZIP file
cd temp_package
zip -r ../webhook_lambda.zip . >/dev/null
cd ..

log "Package created: webhook_lambda.zip"

# Check if function exists
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" >/dev/null 2>&1; then
    log "Updating existing function..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file fileb://webhook_lambda.zip \
        --region "$REGION"
    
    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --timeout "$TIMEOUT" \
        --memory-size "$MEMORY_SIZE" \
        --region "$REGION"
else
    log "Creating new function..."
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime "$RUNTIME" \
        --handler "$HANDLER" \
        --role "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role" \
        --zip-file fileb://webhook_lambda.zip \
        --timeout "$TIMEOUT" \
        --memory-size "$MEMORY_SIZE" \
        --region "$REGION"
fi

# Clean up
rm -rf temp_package webhook_lambda.zip

log "=== Webhook Lambda Deployed Successfully ==="
log "Function: $FUNCTION_NAME"
log "Region: $REGION"
log ""
log "The EC2 script will now use this Lambda for reliable webhook delivery!" 