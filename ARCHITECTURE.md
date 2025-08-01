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
                                        - s3_output: s3://transcription-curevo/results/abc123/transcription_20241201_143022_audio.txt

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

| Function | File | Purpose | Trigger | AMI Dependency |
|----------|------|---------|---------|----------------|
| **TranscriptionAPI** | `lambda_api.py` | Register callback URLs | HTTP API | None |
| **TranscriptionProcessUpload** | `lambda_process_upload.py` | Orchestrate transcription | S3 upload event | Uses AMI_ID env var |
| **TranscriptionWebhookDelivery** | `lambda_webhook_delivery.py` | Send webhook notifications | DynamoDB updates | None |

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

## File Structure & Organization

### Core Transcription System
```
run_transcription/
├── fast_transcribe.sh          # Main EC2 startup script (runs on instance)
├── fast_transcribe.py          # Python transcription engine with pre-compiled state
├── build_ami.sh               # AMI builder with pre-compiled model state creation
└── OPTIMIZATION_SUMMARY.md    # Performance optimization documentation
```

### AWS Infrastructure Setup
```
setup/
├── setup_infrastructure.sh    # Complete AWS resource creation
├── setup_api_gateway.sh       # API Gateway configuration
├── setup_dynamodb.sh          # DynamoDB table creation
├── cleanup_transcription_instances.sh  # Cleanup utility
├── ec2_instance_role_policy.json      # EC2 IAM permissions
└── lambda/
    ├── deploy_lambda_functions.sh     # Lambda deployment with AMI sync
    ├── lambda_process_upload.py       # S3 trigger → EC2 orchestration
    ├── lambda_api.py                  # API endpoint → callback registration
    ├── lambda_webhook_delivery.py     # Webhook notification handler
    ├── bucket_policy.json             # S3 bucket permissions
    ├── lambda_execution_role_policy.json # Lambda IAM permissions
    ├── setup_cloudwatch_logs.sh       # CloudWatch log configuration
    └── verify_lambda_env.sh           # Environment verification
```

### AMI Management & Dependencies
- **Production AMI**: `ami-0398e6a059f8209a3` (optimized with pre-compiled model state, CUDA, dependencies)
- **Build Source**: `ami-0989fb15ce71ba39e` (clean Ubuntu 22.04 LTS used only for AMI construction)
- **Region**: `eu-north-1` (Stockholm)
- **Build Process**: `build_ami.sh` transforms Build Source → Production AMI
- **Dependencies**: Pre-compiled model state, CUDA kernels, optimized Python environment

*The Production AMI is the only one that matters for running transcriptions. The Build Source is just the starting material for creating new Production AMIs.*

### Debugging & Utilities
```
debug_ami_status.sh            # AMI verification and debugging
fix_lambda_permissions.sh      # Permission troubleshooting
```

## AMI Update Workflow

The system uses a coordinated AMI update process to ensure consistency across all components:

### 1. AMI ID Management
- **Hardcoded approach**: AMI IDs are constants in source files [[memory:4672947]]
- **Automatic propagation**: Build script updates deployment scripts
- **Version synchronization**: All components use same AMI ID

### 2. Update Process Flow
```
1. Update source code (fast_transcribe.py, fast_transcribe.sh)
   ↓
2. Run build_ami.sh (creates new AMI + updates Lambda scripts)
   ↓
3. Deploy Lambda functions (deploy_lambda_functions.sh)
   ↓
4. Test transcription (upload audio to S3)
   ↓
5. Verify all components use new AMI
```

### 3. Component Synchronization
- **fast_transcribe.py**: `EXPECTED_AMI_ID` constant (line 23)
- **lambda_process_upload.py**: `AMI_ID` environment variable
- **deploy_lambda_functions.sh**: Hardcoded AMI ID constant (line 11)
- **build_ami.sh**: Automatic sed replacement during AMI creation

### 4. Verification Points
- New AMI ID logged in build output
- Lambda environment variables updated
- EC2 instances launch with correct AMI
- CloudWatch logs show expected AMI ID
- Transcription completes successfully 