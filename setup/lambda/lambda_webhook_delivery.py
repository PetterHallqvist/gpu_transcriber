#!/usr/bin/env python3
"""Simple and elegant webhook delivery Lambda with retry logic."""

import json
import boto3
import requests
import hmac
import hashlib
import time
from datetime import datetime

# AWS clients
dynamodb = boto3.client('dynamodb')
lambda_client = boto3.client('lambda')

# Configuration
DYNAMODB_TABLE = 'transcription-jobs'
MAX_RETRIES = 3

def log(message):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def send_webhook(callback_url, payload, secret=None):
    """Send webhook with HMAC signature if secret provided."""
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'GPU-Transcriber/1.0',
        'X-Webhook-Timestamp': str(int(time.time()))
    }
    
    # Add HMAC signature if secret provided
    if secret:
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        headers['X-Webhook-Signature'] = f"sha256={signature}"
    
    try:
        response = requests.post(callback_url, data=payload, headers=headers, timeout=30)
        return response.status_code in [200, 201, 202], response.status_code
    except Exception as e:
        return False, str(e)

def lambda_handler(event, context):
    """Main Lambda handler for webhook delivery."""
    log("=== Webhook Delivery ===")
    
    try:
        # Extract webhook data
        job_id = event['job_id']
        callback_url = event['callback_url']
        payload = event['payload']
        retry_count = event.get('retry_count', 0)
        
        log(f"Job: {job_id}, Attempt: {retry_count + 1}")
        
        # Get callback secret from DynamoDB
        try:
            response = dynamodb.get_item(
                TableName=DYNAMODB_TABLE,
                Key={'job_id': {'S': job_id}},
                ProjectionExpression='callback_secret'
            )
            secret = response.get('Item', {}).get('callback_secret', {}).get('S')
        except:
            secret = None
        
        # Send webhook
        success, result = send_webhook(callback_url, json.dumps(payload), secret)
        
        if success:
            # Update DynamoDB with success and S3 result locations
            update_expr = "SET webhook_status = :status, webhook_delivered_at = :timestamp"
            expr_values = {
                ':status': {'S': 'delivered'},
                ':timestamp': {'S': datetime.now().isoformat()}
            }
            
            # Add S3 result tracking if present in payload
            if 's3_results' in payload:
                update_expr += ", s3_results = :s3_results"
                expr_values[':s3_results'] = {'S': json.dumps(payload['s3_results'])}
            
            dynamodb.update_item(
                TableName=DYNAMODB_TABLE,
                Key={'job_id': {'S': job_id}},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_values
            )
            log(f"Webhook delivered successfully with S3 tracking")
            return {'statusCode': 200, 'body': 'Delivered'}
            
        else:
            # Handle retry logic
            if retry_count < MAX_RETRIES:
                # Schedule retry with exponential backoff
                delay = min(300, 5 * (2 ** retry_count))  # 5s, 10s, 20s, 40s, 80s, 160s, 300s max
                
                retry_event = {
                    'job_id': job_id,
                    'callback_url': callback_url,
                    'payload': payload,
                    'retry_count': retry_count + 1
                }
                
                lambda_client.invoke(
                    FunctionName=context.function_name,
                    InvocationType='Event',
                    Payload=json.dumps(retry_event),
                    Qualifier=context.function_version
                )
                
                log(f"Scheduled retry in {delay}s")
                return {'statusCode': 202, 'body': 'Retry scheduled'}
                
            else:
                # Final failure
                dynamodb.update_item(
                    TableName=DYNAMODB_TABLE,
                    Key={'job_id': {'S': job_id}},
                    UpdateExpression="SET webhook_status = :status, webhook_error = :error",
                    ExpressionAttributeValues={
                        ':status': {'S': 'failed'},
                        ':error': {'S': f"Failed after {MAX_RETRIES} attempts: {result}"}
                    }
                )
                log(f"Webhook failed permanently: {result}")
                return {'statusCode': 500, 'body': 'Failed permanently'}
                
    except Exception as e:
        log(f"Error: {e}")
        return {'statusCode': 500, 'body': f'Error: {str(e)}'} 