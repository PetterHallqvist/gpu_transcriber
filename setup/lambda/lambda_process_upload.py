#!/usr/bin/env python3
"""Lambda function triggered by S3 uploads to launch EC2 instances for transcription."""

import json
import boto3
import os
import uuid
from datetime import datetime
from botocore.exceptions import ClientError

# AWS clients
ec2 = boto3.client('ec2')
dynamodb = boto3.client('dynamodb')
s3 = boto3.client('s3')

# Configuration
S3_BUCKET = "transcription-curevo"
S3_PREFIX = "transcription_upload/"
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'transcription-jobs')

def log(message):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def ensure_dynamodb_table_exists():
    """Ensure DynamoDB table exists, create if it doesn't."""
    try:
        # Check if table exists
        dynamodb.describe_table(TableName=DYNAMODB_TABLE)
        log(f"DynamoDB table '{DYNAMODB_TABLE}' exists ✓")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            log(f"Creating DynamoDB table '{DYNAMODB_TABLE}'...")
            try:
                dynamodb.create_table(
                    TableName=DYNAMODB_TABLE,
                    AttributeDefinitions=[
                        {'AttributeName': 'job_id', 'AttributeType': 'S'}
                    ],
                    KeySchema=[
                        {'AttributeName': 'job_id', 'KeyType': 'HASH'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )
                
                # Wait for table to be active
                waiter = dynamodb.get_waiter('table_exists')
                waiter.wait(TableName=DYNAMODB_TABLE)
                
                log(f"DynamoDB table '{DYNAMODB_TABLE}' created successfully ✓")
                return True
            except Exception as create_error:
                log(f"Error creating DynamoDB table: {create_error}")
                return False
        else:
            log(f"Error checking DynamoDB table: {e}")
            return False

def create_job_record(job_id, s3_key, status="launching", callback_url=None, callback_secret=None):
    """Create job record in DynamoDB."""
    try:
        # Ensure table exists before creating record
        if not ensure_dynamodb_table_exists():
            log("Failed to ensure DynamoDB table exists")
            return False
            
        item = {
            'job_id': {'S': job_id},
            's3_key': {'S': s3_key},
            'status': {'S': status},
            'created_at': {'S': datetime.now().isoformat()},
            'updated_at': {'S': datetime.now().isoformat()}
        }
        
        # Add callback information if provided
        if callback_url:
            item['callback_url'] = {'S': callback_url}
        if callback_secret:
            item['callback_secret'] = {'S': callback_secret}
            
        dynamodb.put_item(
            TableName=DYNAMODB_TABLE,
            Item=item
        )
        log(f"Created job record: {job_id}")
        return True
    except Exception as e:
        log(f"Error creating job record: {e}")
        return False

def get_working_subnet():
    """Get a working subnet for EC2 instances."""
    # Use environment variable if available, otherwise fallback to hardcoded value
    subnet_id = os.environ.get('SUBNET_ID')
    if subnet_id:
        log(f"Using subnet from environment: {subnet_id}")
        return subnet_id
    else:
        # Fallback to hardcoded subnet for eu-north-1a
        log("Warning: SUBNET_ID not set, using fallback subnet")
        return "subnet-0e18482f4c0b1d4ac"

def create_or_get_security_group():
    """Create or get security group with proper rules for transcription."""
    # Use environment variable if available
    security_group_id = os.environ.get('SECURITY_GROUP_ID')
    if security_group_id:
        log(f"Using security group from environment: {security_group_id}")
        return security_group_id
    
    # Fallback to creating/finding security group by name
    security_group_name = "transcription-g4dn-sg"
    
    try:
        # Try to find existing security group
        response = ec2.describe_security_groups(
            Filters=[{'Name': 'group-name', 'Values': [security_group_name]}]
        )
        
        if response['SecurityGroups']:
            sg_id = response['SecurityGroups'][0]['GroupId']
            log(f"Using existing security group: {sg_id}")
            return sg_id
    except ClientError:
        pass
    
    # Create new security group
    try:
        response = ec2.create_security_group(
            GroupName=security_group_name,
            Description="Security group for GPU transcription instances"
        )
        sg_id = response['GroupId']
        log(f"Created new security group: {sg_id}")
        
        # Add outbound rules for internet access
        ec2.authorize_security_group_egress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    'IpProtocol': '-1',
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        
        # Add inbound SSH rule (optional, for debugging)
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        
        log(f"Security group {sg_id} configured with internet access")
        return sg_id
        
    except ClientError as e:
        log(f"Error creating security group: {e}")
        raise

def launch_ec2_instance(job_id, s3_key):
    """Launch EC2 instance for transcription processing."""
    try:
        # Get working subnet and security group
        subnet_id = get_working_subnet()
        security_group_id = create_or_get_security_group()
        
        # Download and run the fast_transcribe.sh script
        user_data = f"""#!/bin/bash
set -euo pipefail

# Enhanced logging function with CloudWatch integration
log_msg() {{
    local level="$1"
    local message="$2"
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message"
    
    # Send to CloudWatch if available
    if command -v aws &> /dev/null; then
        aws logs put-log-events \
            --log-group-name "/aws/ec2/transcription" \
            --log-stream-name "transcription-{job_id}" \
            --log-events timestamp=$(date +%s)000,message="[$level] $message" \
            --region eu-north-1 2>/dev/null || true
    fi
}}

log_msg "INFO" "=== EC2 Instance Startup ==="

# Wait for instance to be fully ready
log_msg "INFO" "Waiting for instance to be fully ready..."
sleep 30

# Update system packages
log_msg "INFO" "Updating system packages..."
apt-get update -y > /dev/null 2>&1 || log_msg "WARN" "Package update failed"

# Install required packages
log_msg "INFO" "Installing required packages..."
apt-get install -y awscli python3-pip python3-venv ffmpeg > /dev/null 2>&1 || log_msg "WARN" "Package installation failed"

# Create working directory
log_msg "INFO" "Creating working directory..."
mkdir -p /opt/transcription
cd /opt/transcription

# Download the fast_transcribe.sh script from S3
log_msg "INFO" "Downloading fast_transcribe.sh script..."
if aws s3 cp s3://{S3_BUCKET}/scripts/fast_transcribe.sh ./ --region eu-north-1; then
    log_msg "INFO" "Script downloaded successfully"
    chmod +x ./fast_transcribe.sh
else
    log_msg "ERROR" "Failed to download fast_transcribe.sh script"
    exit 1
fi

# Run the fast_transcribe.sh script
log_msg "INFO" "Running fast_transcribe.sh script..."
./fast_transcribe.sh

log_msg "INFO" "=== Instance shutdown complete ==="
"""

        # Launch instance with proper configuration
        response = ec2.run_instances(
            ImageId=os.environ['AMI_ID'],
            InstanceType=os.environ.get('INSTANCE_TYPE', 'g4dn.xlarge'),
            MinCount=1,
            MaxCount=1,
            SecurityGroupIds=[security_group_id],
            SubnetId=subnet_id,
            IamInstanceProfile={'Name': os.environ.get('IAM_ROLE_NAME', 'EC2TranscriptionRole')},  # Use environment variable
            UserData=user_data,
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f'transcription-{job_id}'},
                    {'Key': 'JobId', 'Value': job_id},
                    {'Key': 'Purpose', 'Value': 'transcription'},
                    {'Key': 'AutoTerminate', 'Value': 'true'}
                ]
            }]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        log(f"Launched EC2 instance: {instance_id}")
        log(f"Subnet: {subnet_id}")
        log(f"Security Group: {security_group_id}")
        
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
        
    except ClientError as e:
        log(f"Error launching EC2 instance: {e}")
        raise

def lambda_handler(event, context):
    """Main Lambda handler."""
    log("=== S3 Upload Triggered ===")
    
    try:
        # Extract S3 event
        s3_event = event['Records'][0]['s3']
        bucket_name = s3_event['bucket']['name']
        s3_key = s3_event['object']['key']
        
        log(f"Bucket: {bucket_name}, Key: {s3_key}")
        
        # Validate bucket and prefix
        if bucket_name != S3_BUCKET or not s3_key.startswith(S3_PREFIX):
            log(f"Ignoring upload to {bucket_name}/{s3_key}")
            return {'statusCode': 200, 'body': json.dumps({'message': 'Ignored'})}
        
        # Validate file type
        if not s3_key.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
            log(f"Ignoring non-audio file: {s3_key}")
            return {'statusCode': 200, 'body': json.dumps({'message': 'Not audio file'})}
        
        # Ensure DynamoDB table exists before processing
        log("Ensuring DynamoDB table exists...")
        if not ensure_dynamodb_table_exists():
            log("Failed to ensure DynamoDB table exists")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to setup DynamoDB table'})
            }
        
        # Extract callback information from S3 object metadata or filename
        # For now, we'll use a simple approach: if filename contains callback info
        # In production, you might want to use S3 object metadata or a separate config
        callback_url = None
        callback_secret = None
        
        # Generate new job ID for all uploads (files are now directly in transcription_upload/)
        job_id = str(uuid.uuid4())
        log(f"Generated new job_id: {job_id}")
        
        # Extract filename from S3 key for logging
        filename = s3_key.split('/')[-1] if '/' in s3_key else s3_key
        log(f"Processing file: {filename}")
        
        if not create_job_record(job_id, s3_key, callback_url=callback_url, callback_secret=callback_secret):
            raise Exception("Failed to create job record")
        
        # Launch EC2 instance
        instance_id = launch_ec2_instance(job_id, s3_key)
        
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
        
        log("=== Job Launched Successfully ===")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Transcription job launched',
                'job_id': job_id,
                'instance_id': instance_id,
                's3_key': s3_key
            })
        }
        
    except Exception as e:
        log(f"Error processing S3 event: {e}")
        
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
            'body': json.dumps({'error': 'Failed to process job', 'message': str(e)})
        } 