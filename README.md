# GPU Transcriber - Production Edition

Ultra-fast Swedish transcription with NVIDIA T4 GPU acceleration. **2-3 minute total runtime** with production-grade architecture.

## General Tasks & Architecture Overview

### Core Components Interaction

The system consists of three key files that work together to provide ultra-fast GPU transcription:

#### 1. **build_ami.sh** - AMI Creation & Optimization
- **Purpose**: Creates a pre-optimized Amazon Machine Image (AMI) with all dependencies pre-installed
- **Key Tasks**:
  - Launches a base Ubuntu 22.04 instance
  - Installs CUDA, PyTorch, and Whisper dependencies
  - Downloads and caches the Swedish Whisper model (`KBLab/kb-whisper-small`)
  - Pre-compiles model state for instant loading
  - Creates a production-ready AMI with ID `ami-0862833fe45c7055b`
- **Output**: Optimized AMI that eliminates 90% of startup time

#### 2. **fast_transcribe.sh** - EC2 Instance Orchestrator
- **Purpose**: Main startup script that runs on each EC2 instance during transcription
- **Key Tasks**:
  - Retrieves job metadata from instance tags and DynamoDB
  - Downloads audio file from S3 to local storage
  - Calls `fast_transcribe.py` for actual transcription
  - Uploads results back to S3
  - Updates job status in DynamoDB
  - Terminates the instance when complete
- **Performance**: Starts transcription within 10 seconds of instance launch

#### 3. **fast_transcribe.py** - Core Transcription Engine
- **Purpose**: Python script that performs the actual audio transcription
- **Key Tasks**:
  - Loads pre-cached Whisper model from `/opt/transcribe/models/`
  - Processes audio file using NVIDIA T4 GPU acceleration
  - Generates Swedish transcription with optimized parameters
  - Saves results with metadata (timing, model info)
- **Performance**: Model loads in 3-5 seconds from cache vs 30+ seconds from scratch

### Architecture Flow

```
1. build_ami.sh (One-time setup)
   ↓ Creates optimized AMI
   
2. Client uploads audio to S3
   ↓ Triggers Lambda function
   
3. Lambda launches EC2 with optimized AMI
   ↓ Instance starts with fast_transcribe.sh
   
4. fast_transcribe.sh orchestrates process
   ↓ Downloads audio, calls fast_transcribe.py
   
5. fast_transcribe.py performs transcription
   ↓ Uses pre-cached model for ultra-fast processing
   
6. Results uploaded to S3, instance terminated
```

### Performance Optimization Strategy

| Component | Optimization | Time Saved |
|-----------|-------------|------------|
| **build_ami.sh** | Pre-downloads model to `/opt/transcribe/models/` | 30-45 seconds |
| **fast_transcribe.sh** | Direct S3 download, minimal setup | 10-15 seconds |
| **fast_transcribe.py** | Pre-compiled model state, CUDA warmup | 25-30 seconds |

### Key Integration Points

- **AMI ID Consistency**: All components reference the same AMI ID (`ami-0862833fe45c7055b`)
- **Model Cache Path**: Standardized cache location `/opt/transcribe/models/`
- **Environment Variables**: Shared configuration between shell and Python scripts
- **Error Handling**: Comprehensive logging and status updates across all components

## Performance

| Metric | Time | Cost |
|--------|------|------|
| **Total Runtime** | **2-3 minutes** | **~$0.02** |
| Boot Time | 30-45s | Spot instance |
| Model Load | 3-5s | Pre-compiled state (90% faster) |
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



# 5. Transcribe audio
cd ../run_transcription
./fast_transcribe.sh audio_file.mp3
```

## File Structure & Usage

### Core Transcription Files
```
run_transcription/
├── fast_transcribe.sh          # Main EC2 startup script (runs on instance)
├── fast_transcribe.py          # Python transcription engine with pre-compiled state
├── build_ami.sh               # AMI builder with pre-compiled model state creation
└── OPTIMIZATION_SUMMARY.md    # Performance optimization documentation
```

### Infrastructure Setup
```
setup/
├── setup_infrastructure.sh    # AWS resources (S3, IAM, security groups)

├── cleanup_transcription_instances.sh  # Cleanup utility
├── ec2_instance_role_policy.json      # EC2 permissions
└── lambda/
    ├── deploy_lambda_functions.sh     # Lambda deployment with AMI updates
    ├── lambda_process_upload.py       # S3 trigger handler
    └── lambda_execution_role_policy.json
```

### AMI Management
The system uses a hardcoded AMI ID approach for consistency:
- **Production AMI**: `ami-0398e6a059f8209a3` (optimized with pre-compiled model state)
- **Build Source**: `ami-0989fb15ce71ba39e` (clean Ubuntu 22.04 LTS for building)
- **Region**: `eu-north-1` (Stockholm)

*Note: Only the Production AMI is used for transcription. The Build Source is only used by `build_ami.sh` to create new optimized AMIs.*

## Updating the AMI - Step-by-Step Process

When you need to update the AMI (e.g., for dependency updates, model changes, or optimizations), follow this **exact order**:

### 1. Update AMI References
```bash
# Update fast_transcribe.sh (if needed)
# The script automatically gets AMI ID from instance metadata

# Update fast_transcribe.py
# Edit EXPECTED_AMI_ID constant (line 23)
vim run_transcription/fast_transcribe.py

# Update build_ami.sh
# Edit BUILD_SOURCE constant (line 14) only if changing Ubuntu version
vim run_transcription/build_ami.sh
```

### 2. Build the New AMI
```bash
cd run_transcription
./build_ami.sh
# This will:
# - Launch EC2 instance with base AMI
# - Install all dependencies and models
# - Create new AMI with ID
# - Automatically update Lambda deployment script
# - Log the new AMI ID for next steps
```

### 3. Update Lambda Functions
```bash
cd setup/lambda
# The deploy script will automatically use the new AMI ID
./deploy_lambda_functions.sh
# This updates:
# - Lambda environment variables with new AMI ID
# - Lambda function code
# - IAM permissions
```

### 4. Test the Update
```bash
# Upload audio to S3 to trigger transcription
cd ../run_transcription
./fast_transcribe.sh test_audio.mp3

# Monitor the process:
# - Check CloudWatch logs for Lambda functions
# - Verify EC2 instance launches with new AMI
# - Confirm transcription completes successfully
```

### 5. Verification Checklist
- [ ] New AMI ID is logged in build output
- [ ] Lambda functions deployed successfully
- [ ] EC2 instances launch with new AMI
- [ ] Transcription completes without errors
- [ ] Results are saved to S3 correctly

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

## Lambda Functions

| Function | Purpose | Trigger | AMI Dependency |
|----------|---------|---------|----------------|
| `TranscriptionProcessUpload` | Orchestrate transcription | S3 upload | Uses AMI_ID env var |

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
| Lambda AMI mismatch | Redeploy Lambda functions |

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

# Verify AMI exists
aws ec2 describe-images --image-ids ami-0d090b80bc56081ba --region eu-north-1
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