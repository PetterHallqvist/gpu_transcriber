#!/usr/bin/env python3
"""Lambda function triggered by S3 uploads to launch EC2 instances for transcription."""

import json
import boto3
import os
import uuid
import urllib.parse
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
AMI_ID = 'ami-051be79f88b9cbb42'  # Optimized GPU-enabled AMI for transcription processing

def log(message):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def get_ami_id():
    """Get AMI ID - using predefined constant for reliability."""
    log(f"Using AMI ID: {AMI_ID}")
    return AMI_ID

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

def fix_s3_object_permissions(bucket_name, s3_key):
    """Fix S3 object permissions to ensure EC2 instance can access the file."""
    try:
        log(f"Fixing S3 object permissions for: s3://{bucket_name}/{s3_key}")
        
        # Set object ACL to bucket-owner-full-control to ensure EC2 instance can access
        s3.put_object_acl(
            Bucket=bucket_name,
            Key=s3_key,
            ACL='bucket-owner-full-control'
        )
        
        log(f"S3 object permissions fixed successfully for: s3://{bucket_name}/{s3_key}")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            log(f"Warning: S3 object not found: s3://{bucket_name}/{s3_key}")
            log("This may be due to URL encoding issues - trying to find the actual key...")
            
            # Try to find the actual key by listing objects with similar names
            try:
                prefix = s3_key.rsplit('/', 1)[0] + '/' if '/' in s3_key else ''
                response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        decoded_key = urllib.parse.unquote(obj['Key'])
                        if decoded_key == s3_key:
                            log(f"Found URL-encoded key: {obj['Key']}")
                            # Fix permissions on the actual URL-encoded key
                            s3.put_object_acl(
                                Bucket=bucket_name,
                                Key=obj['Key'],
                                ACL='bucket-owner-full-control'
                            )
                            log(f"Fixed permissions on URL-encoded key: {obj['Key']}")
                            return True
            except Exception as list_error:
                log(f"Error listing objects: {list_error}")
            
            return False
        elif error_code == 'AccessDenied':
            log(f"Warning: Cannot modify S3 object ACL (Access Denied): s3://{bucket_name}/{s3_key}")
            log("This may be due to bucket policy restrictions, but EC2 should still be able to access")
            return True  # Consider this a success since EC2 might still have access
        else:
            log(f"Error fixing S3 object permissions: {e}")
            return False
    except Exception as e:
        log(f"Unexpected error fixing S3 object permissions: {e}")
        return False


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

def launch_ec2_instance(job_id, s3_key, standardized_filename):
    """Launch EC2 instance for transcription processing."""
    try:
        # Get AMI ID first
        ami_id = get_ami_id()
        if not ami_id:
            raise Exception("AMI_ID not available")
        
        # Get working subnet and security group
        subnet_id = get_working_subnet()
        security_group_id = create_or_get_security_group()
        
        # Fast startup script with enhanced error handling and AMI logging
        user_data = f"""#!/bin/bash
set -euo pipefail

# Enhanced logging function
log_msg() {{
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}}

log_msg "Starting transcription instance..."
log_msg "AMI ID: {ami_id}"

# Get and log instance metadata
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
log_msg "Instance ID: $INSTANCE_ID"
log_msg "Instance Type: $INSTANCE_TYPE"

# Verify AMI setup completion marker
if [[ ! -f "/opt/transcribe/.setup_complete" ]]; then
    log_msg "ERROR: AMI setup incomplete - missing /opt/transcribe/.setup_complete"
    log_msg "This indicates the AMI {ami_id} was not properly built"
    exit 1
fi

log_msg "AMI setup verified - proceeding with transcription"

# Setup working directory
mkdir -p /opt/transcription
cd /opt/transcription

# Verify transcription script exists in AMI
if [[ ! -f "/opt/transcribe/fast_transcribe.py" ]]; then
    log_msg "ERROR: Python transcription script missing from AMI"
    log_msg "Expected: /opt/transcribe/fast_transcribe.py"
    exit 1
fi

# Set environment variables for the transcription script
export S3_KEY="{s3_key}"
export STANDARDIZED_FILENAME="{standardized_filename}"
export JOB_ID="{job_id}"

log_msg "Environment variables set:"
log_msg "S3_KEY: $S3_KEY"
log_msg "STANDARDIZED_FILENAME: $STANDARDIZED_FILENAME"
log_msg "JOB_ID: $JOB_ID"

# Check if shell script exists and execute it directly
if [[ -f "/opt/transcription/fast_transcribe.sh" ]]; then
    log_msg "Found transcription shell script at /opt/transcription/fast_transcribe.sh"
    chmod +x /opt/transcription/fast_transcribe.sh
    log_msg "Executing transcription script..."
    /opt/transcription/fast_transcribe.sh
else
    log_msg "ERROR: Shell script not found at /opt/transcription/fast_transcribe.sh"
    log_msg "AMI {ami_id} may be missing required scripts"
    exit 1
fi
"""

        # Launch instance with proper configuration
        log(f"Launching EC2 instance with AMI: {ami_id}")
        log(f"Instance type: {os.environ.get('INSTANCE_TYPE', 'g4dn.xlarge')}")
        log(f"Subnet: {subnet_id}")
        log(f"Security Group: {security_group_id}")
        
        response = ec2.run_instances(
            ImageId=ami_id,
            InstanceType=os.environ.get('INSTANCE_TYPE', 'g4dn.xlarge'),
            MinCount=1,
            MaxCount=1,
            SecurityGroupIds=[security_group_id],
            SubnetId=subnet_id,
            IamInstanceProfile={'Name': os.environ.get('IAM_ROLE_NAME', 'EC2TranscriptionRole')},
            UserData=user_data,
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f'transcription-{job_id}'},
                    {'Key': 'JobId', 'Value': job_id},
                    {'Key': 'Purpose', 'Value': 'transcription'},
                    {'Key': 'AutoTerminate', 'Value': 'true'},
                    {'Key': 'AMI_ID', 'Value': ami_id}  # Tag instance with AMI ID for debugging
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
        original_s3_key = s3_event['object']['key']
        
        # URL-decode the S3 key to handle special characters uploaded via S3 console
        s3_key = urllib.parse.unquote(original_s3_key)
        
        if original_s3_key != s3_key:
            log(f"URL-decoded S3 key: '{original_s3_key}' -> '{s3_key}'")
        
        log(f"Original S3 key: {s3_key}")
        
        # Validate bucket and prefix
        if bucket_name != S3_BUCKET or not s3_key.startswith(S3_PREFIX):
            log(f"Ignoring upload to {bucket_name}/{s3_key}")
            return {'statusCode': 200, 'body': json.dumps({'message': 'Ignored'})}
        
        # Validate file type
        if not s3_key.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
            log(f"Ignoring non-audio file: {s3_key}")
            return {'statusCode': 200, 'body': json.dumps({'message': 'Not audio file'})}
        
        # Generate simple standardized filename: date_time_uuid.extension
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        file_extension = s3_key.split('.')[-1].lower()
        original_filename = s3_key.split('/')[-1] if '/' in s3_key else s3_key
        
        # Create simple standardized filename
        standardized_filename = f"{timestamp}_{file_uuid}.{file_extension}"
        new_s3_key = f"{S3_PREFIX}{standardized_filename}"
        
        log(f"Original filename: {original_filename}")
        log(f"Standardized filename: {standardized_filename}")
        log(f"New S3 key: {new_s3_key}")
        
        # Copy and rename the file in S3 with simple naming
        try:
            log(f"Renaming file from {s3_key} to {new_s3_key}")
            s3.copy_object(
                Bucket=bucket_name,
                CopySource={'Bucket': bucket_name, 'Key': s3_key},
                Key=new_s3_key
            )
            
            # Delete the original file
            s3.delete_object(Bucket=bucket_name, Key=s3_key)
            log(f"Successfully renamed file to: {new_s3_key}")
            
            # Update s3_key to use the new standardized key
            s3_key = new_s3_key
            
        except Exception as e:
            log(f"Error: Failed to rename file: {e}")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': f'Failed to rename file: {str(e)}'})
            }
        
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
        
        # Fix S3 object permissions to ensure EC2 instance can access the file
        # This is critical for files uploaded via S3 console vs CLI
        log("Fixing S3 object permissions...")
        permissions_fixed = fix_s3_object_permissions(bucket_name, s3_key)
        if not permissions_fixed:
            log("Warning: Could not fix S3 object permissions, proceeding anyway")
        
        # Create job record
        if not create_job_record(job_id, s3_key, callback_url=callback_url, callback_secret=callback_secret):
            raise Exception("Failed to create job record")
        
        # Launch EC2 instance with the new standardized S3 key and filename
        instance_id = launch_ec2_instance(job_id, s3_key, standardized_filename)
        
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