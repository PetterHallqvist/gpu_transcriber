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
AMI_ID = 'ami-0cc18c2cbbbc4736c'  # Updated streamlined AMI with pre-compiled model state

def log(message):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def create_job_record(job_id, s3_key):
    """Create job record in DynamoDB."""
    item = {
        'job_id': {'S': job_id},
        's3_key': {'S': s3_key},
        'status': {'S': 'launching'},
        'created_at': {'S': datetime.now().isoformat()},
        'updated_at': {'S': datetime.now().isoformat()}
    }
    
    dynamodb.put_item(TableName=DYNAMODB_TABLE, Item=item)
    log(f"Created job record: {job_id}")

def launch_ec2_instance(job_id, s3_key, standardized_filename):
    """Launch EC2 instance for transcription processing."""
    log(f"Launching EC2 instance with AMI: {AMI_ID}")
    
    # Streamlined user data script
    user_data = f"""#!/bin/bash
set -euo pipefail

log_msg() {{
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}}

log_msg "Starting transcription - AMI: {AMI_ID}"

export S3_KEY="{s3_key}"
export STANDARDIZED_FILENAME="{standardized_filename}"
export JOB_ID="{job_id}"

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
        
        # Simple validation: only process original uploads, not processed files
        filename = s3_key.replace(S3_PREFIX, '')
        if (bucket_name != S3_BUCKET or 
            not s3_key.startswith(S3_PREFIX) or
            not s3_key.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')) or
            filename.split('_')[0].isdigit()):  # Skip timestamped files
            return {'statusCode': 200, 'body': json.dumps({'message': 'Ignored'})}
        
        # Generate standardized filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_uuid = str(uuid.uuid4())[:8]
        file_extension = s3_key.split('.')[-1].lower()
        standardized_filename = f"{timestamp}_{file_uuid}.{file_extension}"
        audiopath = f"{S3_PREFIX}{standardized_filename}"
        
        log(f"Processing: {s3_key} → {audiopath}")
        
        # Fast S3 rename with proper ACL
        s3.copy_object(
            Bucket=bucket_name,
            CopySource={'Bucket': bucket_name, 'Key': s3_key},
            Key=audiopath,
            ACL='bucket-owner-full-control'
        )
        
        # Generate job and create record
        job_id = str(uuid.uuid4())
        create_job_record(job_id, audiopath)
        
        # Launch EC2 instance
        log(f"Launching job: {job_id}")
        instance_id = launch_ec2_instance(job_id, audiopath, standardized_filename)
        
        # Update status to processing
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
        
        log(f"Job launched successfully: {instance_id}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Transcription job launched',
                'job_id': job_id,
                'instance_id': instance_id,
                's3_key': audiopath
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