#!/usr/bin/env python3
"""Streamlined Lambda function for fast transcription processing."""

import json
import boto3
import os
import uuid
import urllib.parse
from datetime import datetime

# AWS clients
ec2 = boto3.client('ec2')
dynamodb = boto3.client('dynamodb')
s3 = boto3.client('s3')

# Configuration
S3_BUCKET = "transcription-curevo"
S3_PREFIX = "transcription_upload/"
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'transcription-jobs')
AMI_ID = os.environ.get('AMI_ID', 'ami-0862833fe45c7055b')  # Read from environment variable

def log(message):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def extract_callback_id(s3_key):
    """Extract callback_id from S3 path if present."""
    # Parse path like: transcription_upload/{callback_id}/filename.mp3
    # or: transcription_upload/filename.mp3 (no callback)
    if not s3_key.startswith(S3_PREFIX):
        return None
    
    # Remove prefix to get relative path
    relative_path = s3_key[len(S3_PREFIX):]
    path_parts = relative_path.split('/')
    
    # If we have more than 1 part, first part might be callback_id
    if len(path_parts) > 1:
        callback_id = path_parts[0]
        # Validate callback_id format (should be a valid UUID or similar)
        if callback_id and len(callback_id) > 0 and not callback_id.startswith('.'):
            # Additional validation: check if it looks like a valid ID
            if len(callback_id) >= 8 and not callback_id.endswith(('.mp3', '.wav', '.m4a', '.flac')):
                return callback_id
    
    return None

def get_callback_details(callback_id):
    """Retrieve callback URL and secret from DynamoDB."""
    try:
        # Query DynamoDB for callback record
        response = dynamodb.get_item(
            TableName=DYNAMODB_TABLE,
            Key={'job_id': {'S': callback_id}}
        )
        
        if 'Item' in response:
            item = response['Item']
            callback_url = item.get('callback_url', {}).get('S')
            callback_secret = item.get('callback_secret', {}).get('S')
            
            if callback_url:
                return {
                    'callback_url': callback_url,
                    'callback_secret': callback_secret
                }
        
        log(f"No callback details found for callback_id: {callback_id}")
        return None
        
    except Exception as e:
        log(f"Error retrieving callback details: {e}")
        return None

def register_callback(callback_id, callback_url, callback_secret=None):
    """Register a callback for future use."""
    try:
        item = {
            'job_id': {'S': callback_id},
            'callback_url': {'S': callback_url},
            'status': {'S': 'registered'},
            'created_at': {'S': datetime.now().isoformat()},
            'updated_at': {'S': datetime.now().isoformat()}
        }
        
        if callback_secret:
            item['callback_secret'] = {'S': callback_secret}
        
        dynamodb.put_item(TableName=DYNAMODB_TABLE, Item=item)
        log(f"Registered callback: {callback_id}")
        return True
        
    except Exception as e:
        log(f"Error registering callback: {e}")
        return False

def create_job_record(job_id, s3_key, callback_info=None):
    """Enhanced to include callback information."""
    item = {
        'job_id': {'S': job_id},
        's3_key': {'S': s3_key},
        'status': {'S': 'launching'},
        'created_at': {'S': datetime.now().isoformat()},
        'updated_at': {'S': datetime.now().isoformat()}
    }
    
    # Add callback information if provided
    if callback_info:
        if callback_info.get('callback_id'):
            item['callback_id'] = {'S': callback_info['callback_id']}
        if callback_info.get('callback_url'):
            item['callback_url'] = {'S': callback_info['callback_url']}
        if callback_info.get('callback_secret'):
            item['callback_secret'] = {'S': callback_info['callback_secret']}
    
    dynamodb.put_item(TableName=DYNAMODB_TABLE, Item=item)
    log(f"Created job record: {job_id}")

def launch_ec2_instance(job_id, s3_key, standardized_filename, callback_info=None):
    """Enhanced to pass callback info to EC2 instance."""
    log(f"Launching EC2 instance with AMI: {AMI_ID}")
    
    # Build environment variables for callback
    callback_env_vars = ""
    if callback_info:
        if callback_info.get('callback_id'):
            callback_env_vars += f'export CALLBACK_ID="{callback_info["callback_id"]}"\n'
        if callback_info.get('callback_url'):
            callback_env_vars += f'export CALLBACK_URL="{callback_info["callback_url"]}"\n'
        if callback_info.get('callback_secret'):
            callback_env_vars += f'export CALLBACK_SECRET="{callback_info["callback_secret"]}"\n'
    
    # Enhanced user data script with callback support
    user_data = f"""#!/bin/bash
set -euo pipefail

log_msg() {{
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}}

log_msg "Starting transcription - AMI: {AMI_ID}"

export S3_KEY="{s3_key}"
export STANDARDIZED_FILENAME="{standardized_filename}"
export JOB_ID="{job_id}"

{callback_env_vars}
log_msg "Starting transcription: $JOB_ID"
chmod +x /opt/transcribe/fast_transcribe.sh
/opt/transcribe/fast_transcribe.sh
"""

    response = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=os.environ.get('INSTANCE_TYPE', 'g4dn.xlarge'),
        MinCount=1,
        MaxCount=1,
        SecurityGroupIds=[os.environ['SECURITY_GROUP_ID']],
        SubnetId=os.environ['SUBNET_ID'],
        IamInstanceProfile={'Name': os.environ.get('IAM_ROLE_NAME', 'EC2TranscriptionRole')},
        UserData=user_data,
        TagSpecifications=[{
            'ResourceType': 'instance',
            'Tags': [
                {'Key': 'Name', 'Value': f'transcription-{job_id}'},
                {'Key': 'JobId', 'Value': job_id},
                {'Key': 'AutoTerminate', 'Value': 'true'}
            ]
        }]
    )
    
    instance_id = response['Instances'][0]['InstanceId']
    log(f"Launched instance: {instance_id}")
    
    # Update DynamoDB with instance ID
    dynamodb.update_item(
        TableName=DYNAMODB_TABLE,
        Key={'job_id': {'S': job_id}},
        UpdateExpression="SET instance_id = :instance_id, updated_at = :updated_at",
        ExpressionAttributeValues={
            ':instance_id': {'S': instance_id},
            ':updated_at': {'S': datetime.now().isoformat()}
        }
    )
    
    return instance_id

def lambda_handler(event, context):
    """Streamlined Lambda handler for fast transcription processing."""
    log("S3 Upload Triggered")
    
    try:
        # Extract and decode S3 event
        s3_event = event['Records'][0]['s3']
        bucket_name = s3_event['bucket']['name']
        s3_key = urllib.parse.unquote(s3_event['object']['key'])
        
        # Enhanced path validation to handle both direct uploads and callback-associated uploads
        if bucket_name != S3_BUCKET:
            log(f"Invalid bucket: {bucket_name}")
            return {'statusCode': 200, 'body': json.dumps({'message': 'Invalid bucket'})}
        
        if not s3_key.startswith(S3_PREFIX):
            log(f"Invalid S3 prefix: {s3_key}")
            return {'statusCode': 200, 'body': json.dumps({'message': 'Invalid S3 prefix'})}
        
        if not s3_key.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
            log(f"Invalid file type: {s3_key}")
            return {'statusCode': 200, 'body': json.dumps({'message': 'Invalid file type'})}
        
        # Extract callback_id from S3 path if present
        callback_id = extract_callback_id(s3_key)
        callback_info = None
        
        if callback_id:
            log(f"Found callback_id in path: {callback_id}")
            try:
                callback_details = get_callback_details(callback_id)
                if callback_details:
                    callback_info = {
                        'callback_id': callback_id,
                        'callback_url': callback_details['callback_url'],
                        'callback_secret': callback_details.get('callback_secret')
                    }
                    log(f"Retrieved callback details for: {callback_id}")
                else:
                    log(f"No callback details found for: {callback_id}, proceeding without callback")
            except Exception as e:
                log(f"Error retrieving callback details: {e}, proceeding without callback")
        else:
            log("No callback_id found in path, proceeding without callback")
        
        # Generate standardized filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_uuid = str(uuid.uuid4())[:8]
        file_extension = s3_key.split('.')[-1].lower()
        standardized_filename = f"{timestamp}_{file_uuid}.{file_extension}"
        audiopath = f"{S3_PREFIX}{standardized_filename}"
        
        log(f"Processing: {s3_key} → {audiopath}")
        
        # Fast S3 rename with proper ACL
        try:
            s3.copy_object(
                Bucket=bucket_name,
                CopySource={'Bucket': bucket_name, 'Key': s3_key},
                Key=audiopath,
                ACL='bucket-owner-full-control'
            )
            log(f"S3 copy successful: {s3_key} → {audiopath}")
        except Exception as e:
            log(f"Error copying S3 object: {e}")
            return {'statusCode': 500, 'body': json.dumps({'error': 'S3 copy failed'})}
        
        # Generate job and create record with callback information
        job_id = str(uuid.uuid4())
        try:
            create_job_record(job_id, audiopath, callback_info)
        except Exception as e:
            log(f"Error creating job record: {e}")
            return {'statusCode': 500, 'body': json.dumps({'error': 'Failed to create job record'})}
        
        # Launch EC2 instance with callback information
        log(f"Launching job: {job_id}")
        try:
            instance_id = launch_ec2_instance(job_id, audiopath, standardized_filename, callback_info)
        except Exception as e:
            log(f"Error launching EC2 instance: {e}")
            # Update job status to failed
            try:
                dynamodb.update_item(
                    TableName=DYNAMODB_TABLE,
                    Key={'job_id': {'S': job_id}},
                    UpdateExpression="SET #status = :status, updated_at = :updated_at",
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={
                        ':status': {'S': 'failed'},
                        ':updated_at': {'S': datetime.now().isoformat()}
                    }
                )
            except:
                pass
            return {'statusCode': 500, 'body': json.dumps({'error': 'Failed to launch EC2 instance'})}
        
        # Update status to processing
        try:
            dynamodb.update_item(
                TableName=DYNAMODB_TABLE,
                Key={'job_id': {'S': job_id}},
                UpdateExpression="SET #status = :status, updated_at = :updated_at",
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':status': {'S': 'processing'},
                    ':updated_at': {'S': datetime.now().isoformat()}
                }
            )
        except Exception as e:
            log(f"Warning: Failed to update job status: {e}")
        
        log(f"✓ Job launched successfully: {job_id} (instance: {instance_id})")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Transcription job launched',
                'job_id': job_id,
                'instance_id': instance_id,
                's3_key': audiopath,
                'callback_id': callback_info.get('callback_id') if callback_info else None
            })
        }
        
    except Exception as e:
        log(f"Error: {e}")
        
        # Update job status to failed if job_id exists
        if 'job_id' in locals():
            try:
                dynamodb.update_item(
                    TableName=DYNAMODB_TABLE,
                    Key={'job_id': {'S': job_id}},
                    UpdateExpression="SET #status = :status, error_message = :error, updated_at = :updated_at",
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={
                        ':status': {'S': 'failed'},
                        ':error': {'S': str(e)},
                        ':updated_at': {'S': datetime.now().isoformat()}
                    }
                )
            except:
                pass
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        } 