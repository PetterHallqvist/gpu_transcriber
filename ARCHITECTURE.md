# GPU Transcription System Architecture

## Overview

This document describes the **ultra-simplified architecture** of the GPU-powered transcription system that provides ultra-fast Swedish audio transcription using NVIDIA T4 GPUs on AWS EC2 instances. The system uses only **2 Lambda functions** and relies entirely on webhooks for client notifications.

## System Flow

### 1. Register Callback (Optional)

```
Client --------------------------> API Gateway --------------------------> Lambda (lambda_api.py)
        POST /api                           (Auth via API Key)
        X-API-Key: client-key-123
        { 
          "callback_url": "https://client.app/webhook",
          "callback_secret": "shared-secret-456"  // optional for HMAC
        }
                                                                              |
                                                                              ↓
                                                                          DynamoDB
                                                                    (transcription-jobs table)
<-------------------------------------------------------------------- 200 OK
        {
          "callback_id": "abc123",
          "upload_path": "transcription_upload/abc123/",
          "message": "Upload audio files to the provided S3 path"
        }
```

### 2. Direct S3 Upload (Triggers Everything)

```
Client --------------------------> S3 (Direct Upload)
        aws s3 cp audio.mp3 s3://transcription-curevo/transcription_upload/abc123/audio.mp3
```

### 3. S3 Event Triggers Complete Process

```
S3 -----> Lambda (lambda_process_upload.py) -----> EC2 Instance Launch
   S3 Event                              with UserData:
                                        - job_id: abc123
                                        - s3_input: s3://transcription-curevo/transcription_upload/abc123/audio.mp3
                                        - s3_output: s3://transcription-curevo/results/abc123/

                                   [2-3 minutes processing]
                                            ↓
EC2 Worker -----> DynamoDB (get callback info)
            |              ↓
            |         callback_url & secret
            |              ↓
            |-----> Direct Webhook POST -----------------------> Client
            |       POST https://client.app/webhook
            |       X-Webhook-Signature: hmac-sha256=...
            |       {
            |         "job_id": "abc123",
            |         "status": "complete",
            |         "transcript": {
            |           "text": "Actual transcription text...",
            |           "segments": []
            |         },
            |         "metadata": {
            |           "processing_time": "120s",
            |           "file_size": "1024000",
            |           "model": "kb-whisper-small"
            |         }
            |       }
            |
            |-----> S3 (async upload for storage)
                    s3://transcription-curevo/results/abc123/transcription_20241201_143022_audio.txt
                    (happens after webhook sent)
```

## Component Details

### AWS Services

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **S3 Bucket** | `transcription-curevo` | File storage with event triggers |
| **DynamoDB Table** | `transcription-jobs` | Job metadata and status tracking |
| **EC2 Instance** | `g4dn.xlarge` | NVIDIA T4 GPU for transcription |
| **Lambda Functions** | Serverless orchestration | Event-driven processing |
| **API Gateway** | REST API endpoints | Client interface |

### Lambda Functions

| Function | File | Purpose | Trigger |
|----------|------|---------|---------|
| **TranscriptionAPI** | `lambda_api.py` | Register callback URLs | HTTP API |
| **TranscriptionProcessUpload** | `lambda_process_upload.py` | Orchestrate transcription | S3 upload event |

### S3 Bucket Structure

```
transcription-curevo/
├── transcription_upload/          # Client uploads
│   ├── 20241201_143022_audio.mp3
│   └── ...
├── results/                       # Transcription outputs
│   ├── abc123/
│   │   └── transcription_20241201_143022_audio.txt
│   └── ...
└── scripts/                       # EC2 worker scripts
    └── fast_transcribe.py
```

### DynamoDB Schema

```json
{
  "job_id": "abc123",
  "status": "registered|launching|processing|completed|failed",
  "s3_key": "transcription_upload/abc123/audio.mp3",
  "callback_url": "https://client.app/webhook",
  "callback_secret": "shared-secret-456",
  "created_at": "2024-12-01T14:30:22Z",
  "updated_at": "2024-12-01T14:32:45Z",
  "instance_id": "i-1234567890abcdef0",
  "error_message": "Error details if failed"
}
```

## Performance Characteristics

| Metric | Time | Cost |
|--------|------|------|
| **Total Runtime** | **2-3 minutes** | **~$0.02** |
| Boot Time | 30-45s | Spot instance |
| Model Load | 5-10s | Pre-cached |
| Transcription | 90-120s | T4 GPU |
| Cleanup | 15-30s | Auto-terminate |

## Security Features

- **API Key Authentication** for client requests
- **Direct S3 Uploads** with IAM-based access control
- **HMAC Webhook Signatures** for callback verification
- **IAM Roles** with least-privilege access
- **S3 Bucket Policies** for access control
- **Encrypted Storage** (AES256)

## Error Handling

- **Automatic Instance Termination** on completion/failure
- **DynamoDB Status Updates** for job tracking
- **CloudWatch Logging** for monitoring
- **Graceful Fallbacks** for failed operations
- **Manual Cleanup Utilities** for orphaned instances

## Cost Optimization

- **Spot Instances**: ~70% savings over on-demand
- **Pay-per-request DynamoDB**: Auto-scaling
- **Automatic Termination**: No idle instance costs
- **Efficient Resource Usage**: GPU-optimized processing

## Monitoring & Logging

- **CloudWatch Logs** for all Lambda functions
- **EC2 Instance Logs** for transcription process
- **DynamoDB Metrics** for job tracking
- **S3 Access Logs** for file operations
- **API Gateway Logs** for client requests

## Deployment Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │    │   API Gateway    │    │   Lambda        │
│                 │───▶│   (REST API)     │───▶│   Functions     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   S3 Bucket     │◀───│   DynamoDB       │◀───│   EC2 Instance  │
│   (Storage)     │    │   (Metadata)     │    │   (GPU Worker)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## File Locations

### Lambda Functions
- `lambda/lambda_api.py` - Callback registration
- `lambda/lambda_process_upload.py` - EC2 instance orchestration

### Infrastructure Setup
- `setup/setup_infrastructure.sh` - AWS resource creation
- `setup/lambda/deploy_lambda_functions.sh` - Lambda deployment
- `setup/setup_api_gateway.sh` - API Gateway configuration

### Worker Scripts
- `run_transcription/fast_transcribe.py` - GPU transcription engine
- `run_transcription/build_ami.sh` - AMI creation script

### Configuration Files
- `setup/lambda/bucket_policy.json` - S3 bucket permissions
- `setup/ec2_instance_role_policy.json` - EC2 IAM role 