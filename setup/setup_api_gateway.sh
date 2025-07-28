#!/bin/bash
# Simplified API Gateway Setup - Unified Transcription Flow
# Supports the ultra-simplified architecture with webhook-based notifications

set -e

REGION="eu-north-1"
API_NAME="TranscriptionAPI"
STAGE="prod"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check dependencies
if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI is required. Please install it first."
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required. Install with: brew install jq (macOS) or sudo apt-get install jq (Ubuntu)"
    exit 1
fi

# Check if Lambda function exists
log "Verifying Lambda function exists..."
if ! aws lambda get-function --function-name TranscriptionAPI --region $REGION &>/dev/null; then
    echo "ERROR: Lambda function 'TranscriptionAPI' not found in region $REGION"
    echo "Please deploy Lambda functions first: cd setup/lambda && ./deploy_lambda_functions.sh"
    exit 1
fi

log "=== Setting up Simplified API Gateway ==="

# Get or create API Gateway
API_ID=$(aws apigateway get-rest-apis --region $REGION --query "items[?name=='$API_NAME'].id" --output text 2>/dev/null || echo "")
if [[ -z "$API_ID" || "$API_ID" == "None" ]]; then
    log "Creating API Gateway..."
    API_ID=$(aws apigateway create-rest-api --name $API_NAME --description "Simplified transcription service with webhook notifications" --region $REGION --query 'id' --output text)
    log "Created API Gateway with ID: $API_ID"
else
    log "Using existing API Gateway with ID: $API_ID"
fi

ROOT_ID=$(aws apigateway get-resources --rest-api-id $API_ID --region $REGION --query 'items[?path==`/`].id' --output text)

# Setup single endpoint for all API operations
log "Setting up /api endpoint..."

# Create resource
RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --region $REGION --query "items[?path=='api'].id" --output text 2>/dev/null || echo "")
if [[ -z "$RESOURCE_ID" || "$RESOURCE_ID" == "None" ]]; then
    RESOURCE_ID=$(aws apigateway create-resource --rest-api-id $API_ID --parent-id $ROOT_ID --path-part api --region $REGION --query 'id' --output text)
    log "Created /api resource with ID: $RESOURCE_ID"
else
    log "Using existing /api resource with ID: $RESOURCE_ID"
fi

# Setup POST method with API key required
log "Configuring POST method with API key authentication..."
aws apigateway put-method --rest-api-id $API_ID --resource-id $RESOURCE_ID --http-method POST --authorization-type NONE --api-key-required true --region $REGION

# Lambda integration
log "Setting up Lambda integration..."
LAMBDA_ARN=$(aws lambda get-function --function-name TranscriptionAPI --region $REGION --query 'Configuration.FunctionArn' --output text)
aws apigateway put-integration --rest-api-id $API_ID --resource-id $RESOURCE_ID --http-method POST --type AWS_PROXY --integration-http-method POST --uri arn:aws:apigateway:$REGION:lambda:path/2015-03-31/functions/$LAMBDA_ARN/invocations --region $REGION

# Add CORS support
log "Adding CORS support..."
aws apigateway put-method --rest-api-id $API_ID --resource-id $RESOURCE_ID --http-method OPTIONS --authorization-type NONE --region $REGION
aws apigateway put-integration --rest-api-id $API_ID --resource-id $RESOURCE_ID --http-method OPTIONS --type MOCK --request-templates '{"application/json":"{\"statusCode\": 200}"}' --region $REGION

# CORS headers
aws apigateway put-method-response --rest-api-id $API_ID --resource-id $RESOURCE_ID --http-method OPTIONS --status-code 200 --response-parameters '{"method.response.header.Access-Control-Allow-Headers":true,"method.response.header.Access-Control-Allow-Methods":true,"method.response.header.Access-Control-Allow-Origin":true}' --region $REGION
aws apigateway put-integration-response --rest-api-id $API_ID --resource-id $RESOURCE_ID --http-method OPTIONS --status-code 200 --response-parameters '{"method.response.header.Access-Control-Allow-Headers":"'\''Content-Type,X-Api-Key'\''","method.response.header.Access-Control-Allow-Methods":"'\''POST,OPTIONS'\''","method.response.header.Access-Control-Allow-Origin":"'\''*'\''"}' --region $REGION

# Create API key and usage plan
log "Setting up API key authentication..."
API_KEY_RESPONSE=$(aws apigateway create-api-key --name TranscriptionAPIKey --enabled true --region $REGION 2>/dev/null || aws apigateway get-api-keys --name-query TranscriptionAPIKey --region $REGION --query 'items[0]' --output json)
API_KEY_VALUE=$(echo $API_KEY_RESPONSE | jq -r '.value')

if [[ "$API_KEY_VALUE" == "null" || -z "$API_KEY_VALUE" ]]; then
    log "Creating new API key..."
    API_KEY_RESPONSE=$(aws apigateway create-api-key --name TranscriptionAPIKey --enabled true --region $REGION)
    API_KEY_VALUE=$(echo $API_KEY_RESPONSE | jq -r '.value')
fi

USAGE_PLAN_ID=$(aws apigateway create-usage-plan --name TranscriptionUsagePlan --api-stages apiId=$API_ID,stage=$STAGE --throttle burstLimit=100,rateLimit=50 --region $REGION --query 'id' --output text 2>/dev/null || aws apigateway get-usage-plans --name-query TranscriptionUsagePlan --region $REGION --query 'items[0].id' --output text)
aws apigateway create-usage-plan-key --usage-plan-id $USAGE_PLAN_ID --key-id $(echo $API_KEY_RESPONSE | jq -r '.id') --key-type API_KEY --region $REGION 2>/dev/null || true

# Deploy and add permissions
log "Deploying API..."
aws apigateway create-deployment --rest-api-id $API_ID --stage-name $STAGE --region $REGION

log "Adding Lambda permissions..."
aws lambda add-permission --function-name TranscriptionAPI --statement-id APIGatewayInvoke --action lambda:InvokeFunction --principal apigateway.amazonaws.com --source-arn "arn:aws:execute-api:$REGION:$(aws sts get-caller-identity --query Account --output text):$API_ID/*" --region $REGION 2>/dev/null || true

API_URL="https://$API_ID.execute-api.$REGION.amazonaws.com/$STAGE"

log "=== Simplified API Gateway Setup Complete! ==="
log "API URL: $API_URL"
log "API Key: $API_KEY_VALUE"
log ""
log "Endpoint:"
log "  POST $API_URL/api  - Register callback URL for webhook notifications"
log ""
log "Usage:"
log "1. Register callback:"
log "curl -X POST $API_URL/api \\"
log "  -H 'Content-Type: application/json' \\"
log "  -H 'x-api-key: $API_KEY_VALUE' \\"
log "  -d '{\"callback_url\": \"https://your-app.com/webhook\"}'"
log ""
log "2. Upload audio to S3:"
log "aws s3 cp audio.mp3 s3://transcription-curevo/transcription_upload/{callback_id}/audio.mp3"
log ""
log "3. Wait for webhook notification:"
log "POST https://your-app.com/webhook"
log "{\"job_id\": \"abc123\", \"status\": \"complete\", \"transcript\": {...}}"
log ""
log "Note: No status checking needed - webhook provides all necessary information!"
log ""
log "Architecture: Client → API Gateway → Lambda → S3 → EC2 (T4 GPU) → Webhook" 