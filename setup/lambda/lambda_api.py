#!/usr/bin/env python3
"""Lambda function to register callback URLs for transcription jobs."""

import json
import boto3
import os
import uuid
from datetime import datetime

# AWS clients
dynamodb = boto3.client('dynamodb')

# Configuration
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'transcription-jobs')

def log(message):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def create_callback_record(callback_id, callback_url, callback_secret=None):
    """Create callback record in DynamoDB."""
    try:
        item = {
            'job_id': {'S': callback_id},  # Use callback_id as job_id for consistency
            'callback_url': {'S': callback_url},
            'status': {'S': 'registered'},  # Initial status
            'created_at': {'S': datetime.now().isoformat()},
            'updated_at': {'S': datetime.now().isoformat()}
        }
        
        if callback_secret:
            item['callback_secret'] = {'S': callback_secret}
        
        dynamodb.put_item(
            TableName=DYNAMODB_TABLE,
            Item=item
        )
        
        log(f"Created callback record: {callback_id}")
        return True
        
    except Exception as e:
        log(f"Error creating callback record: {e}")
        return False

def lambda_handler(event, context):
    """Main Lambda handler."""
    log("=== Register Callback URL ===")
    
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}')) if event.get('body') else event
        
        # Extract parameters
        callback_url = body.get('callback_url')
        callback_secret = body.get('callback_secret')  # optional
        
        if not callback_url:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Missing callback_url parameter'})
            }
        
        # Validate callback URL format
        if not callback_url.startswith(('http://', 'https://')):
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Invalid callback_url format'})
            }
        
        # Generate unique callback ID
        callback_id = str(uuid.uuid4())
        log(f"Generated callback ID: {callback_id}")
        
        # Create callback record in DynamoDB
        if not create_callback_record(callback_id, callback_url, callback_secret):
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Failed to create callback record'})
            }
        
        # Prepare response
        response_data = {
            'callback_id': callback_id,
            'callback_url': callback_url,
            'upload_path': f"transcription_upload/{callback_id}/",
            'message': 'Upload audio files to the provided S3 path to trigger transcription'
        }
        
        log("=== Callback Registration Complete ===")
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(response_data)
        }
        
    except json.JSONDecodeError as e:
        log(f"Invalid JSON in request body: {e}")
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Invalid JSON in request body'})
        }
        
    except Exception as e:
        log(f"Error registering callback: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Failed to register callback', 'message': str(e)})
        } 