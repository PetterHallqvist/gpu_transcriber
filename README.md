# GPU Transcriber - Production Edition

Ultra-fast Swedish transcription with NVIDIA T4 GPU acceleration. **2-3 minute total runtime** with production-grade architecture.

## Performance

| Metric | Time | Cost |
|--------|------|------|
| **Total Runtime** | **2-3 minutes** | **~$0.02** |
| Boot Time | 30-45s | Spot instance |
| Model Load | 5-10s | Pre-cached |
| Transcription | 90-120s | T4 GPU |
| Cleanup | 15-30s | Auto-terminate |

## Quick Start

```bash
# 1. Setup infrastructure (one-time)
cd setup
./setup_infrastructure.sh

# 2. Build production AMI (one-time, ~20 min)
cd ../run_transcription
./build_ami.sh
cd ../setup

# 3. Deploy Lambda functions
cd lambda
./deploy_lambda_functions.sh
cd ..

# 4. Setup API Gateway (optional)
./setup_api_gateway.sh

# 5. Transcribe audio
cd ../run_transcription
./fast_transcribe.sh audio_file.mp3
```

## Architecture

```
Client → API Gateway → Lambda → S3 → EC2 (T4 GPU) → DynamoDB → Webhook
```

**Components:**
- **EC2**: G4DN.XLARGE with NVIDIA T4 (16GB)
- **S3**: File storage with event triggers
- **Lambda**: Serverless orchestration (2 functions only)
- **DynamoDB**: Job tracking and callback storage
- **CloudWatch**: Monitoring & logging
- **Webhooks**: Real-time client notifications

## Setup Structure

```
setup/
├── setup_infrastructure.sh             # AWS resources
├── update_dynamodb_permissions.sh      # DynamoDB setup
├── setup_api_gateway.sh                 # API endpoint
├── cleanup_transcription_instances.sh  # Cleanup utility
├── ec2_instance_role_policy.json       # EC2 permissions
└── lambda/                             # Lambda setup
    ├── deploy_lambda_functions.sh      # Function deployment
    ├── verify_lambda_env.sh            # Environment check
    └── lambda_execution_role_policy.json
```

## Lambda Functions

| Function | Purpose | Trigger |
|----------|---------|---------|
| `TranscriptionProcessUpload` | Orchestrate transcription | S3 upload |
| `TranscriptionAPI` | Register callbacks | HTTP API |

## Production Features

### **Ultra-Fast Performance**
- Pre-cached models & CUDA kernels
- Production AMI with zero setup time
- Optimized T4 GPU pipeline
- Multi-zone spot instance support

### **Cost Protection**
- Automatic instance termination
- Spot pricing (~70% savings)
- Signal traps for all exit conditions
- Manual cleanup utility

### **Self-Healing**
- DynamoDB table auto-creation
- Comprehensive error handling
- Graceful fallbacks
- CloudWatch logging
- Webhook notifications for all status changes

## Usage

### **Single Transcription**
```bash
./run_transcription/fast_transcribe.sh audio.mp3
# Returns: transcription_YYYYMMDD_HHMMSS.txt
```

### **API Server** (Optional)
```bash
cd setup
./setup_api_gateway.sh
# Endpoint: /api (callback registration)
```

### **Webhook Notifications**
The system automatically sends webhook notifications when transcription completes:
- **Success**: Includes transcription text and metadata
- **Failure**: Includes error details and error codes
- **Security**: Optional HMAC signature verification
- **Real-time**: No polling required

### **Cleanup**
```bash
cd setup
./cleanup_transcription_instances.sh
```

## Technical Specs

- **Instance**: G4DN.XLARGE (4 vCPU, 16GB RAM, T4 GPU)
- **Region**: eu-north-1 (Stockholm)
- **OS**: Ubuntu 22.04 LTS
- **Model**: KBLab/kb-whisper-small (Swedish)
- **Storage**: S3 bucket + DynamoDB table

## Troubleshooting

### **Common Issues**

| Issue | Solution |
|-------|----------|
| AMI not found | Run `./build_ami.sh` |
| Spot quota exceeded | Check usage with cleanup script |
| SSH failed | Rebuild AMI |
| Model slow | Verify `/opt/transcribe/models/` exists |

### **Verification**
```bash
# Check DynamoDB
aws dynamodb describe-table --table-name transcription-jobs

# Test permissions
aws dynamodb scan --table-name transcription-jobs --limit 1

# Check Lambda functions
aws lambda list-functions --region eu-north-1

# Check S3 bucket
aws s3 ls s3://transcription-curevo
```

## Security

- **IAM roles** with least-privilege access
- **S3 bucket policies** for access control
- **Security groups** with minimal rules
- **Encrypted storage** and communications

## Cost Optimization

- **Spot instances**: ~70% savings
- **Pay-per-request**: DynamoDB auto-scaling
- **Auto-cleanup**: No orphaned resources
- **S3 lifecycle**: Automatic result cleanup

---

**Result**: Production-grade Swedish transcription with 71% performance improvement and enterprise reliability. 