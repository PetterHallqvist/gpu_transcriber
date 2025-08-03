# GPU Transcription System Architecture

## Overview

This document describes the **ultra-simplified architecture** of the GPU-powered transcription system that provides ultra-fast Swedish audio transcription using NVIDIA T4 GPUs on AWS EC2 instances. The system uses only **1 Lambda function** and relies on direct S3 uploads and downloads.

## System Flow

### Direct S3 Upload (Triggers Everything)

```
Client --------------------------> S3 (Direct Upload)
        aws s3 cp audio.mp3 s3://transcription-curevo/transcription_upload/audio.mp3
```

### S3 Event Triggers Complete Process

```
S3 -----> Lambda (lambda_process_upload.py) -----> EC2 Instance Launch
   S3 Event                              with UserData:
                                        - standardized_filename: 20241201_143022_abc123.mp3
                                        - s3_input: s3://transcription-curevo/transcription_upload/20241201_143022_abc123.mp3
                                        - s3_output: s3://transcription-curevo/results/20241201_143022_abc123.txt

                                   [2-3 minutes processing]
                                            ↓
EC2 Worker -----> S3 (upload results)
            |       s3://transcription-curevo/results/20241201_143022_abc123.txt
            |
            |-----> S3 (upload results)
                    s3://transcription-curevo/results/20241201_143022_abc123.txt
```

## Component Details

### AWS Services

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **S3 Bucket** | `transcription-curevo` | File storage with event triggers |
| **EC2 Instance** | `g4dn.xlarge` | NVIDIA T4 GPU for transcription |
| **Lambda Function** | Serverless orchestration | Event-driven processing |

### Lambda Functions

| Function | File | Purpose | Trigger | AMI Dependency |
|----------|------|---------|---------|----------------|
| **TranscriptionProcessUpload** | `lambda_process_upload.py` | Orchestrate transcription | S3 upload event | Uses AMI_ID env var |

### S3 Bucket Structure

```
transcription-curevo/
├── transcription_upload/          # Client uploads
│   ├── audio.mp3
│   └── ...
├── results/                       # Transcription outputs
│   ├── 20241201_143022_abc123.txt
│   └── ...
└── scripts/                       # EC2 worker scripts
    └── fast_transcribe.py
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

- **Direct S3 Uploads** with IAM-based access control
- **EC2 Instance Profiles** for secure AWS service access
- **Auto-terminating instances** for cost control

## Usage

### 1. Upload Audio File
```bash
aws s3 cp audio.mp3 s3://transcription-curevo/transcription_upload/audio.mp3
```

### 2. Wait for Processing
The system automatically:
- Detects the upload
- Launches an EC2 instance with T4 GPU
- Processes the audio file
- Uploads results to S3
- Terminates the instance

### 3. Download Results
```bash
aws s3 ls s3://transcription-curevo/results/
aws s3 cp s3://transcription-curevo/results/20241201_143022_abc123.txt .
```

## Architecture Benefits

- **Ultra-simple**: Only 1 Lambda function
- **Cost-effective**: Pay only for processing time
- **Scalable**: Automatic instance management
- **Fast**: GPU-accelerated transcription
- **Reliable**: Direct S3 integration

## File Structure

```
gpu_transcriber/
├── setup/
│   ├── lambda/
│   │   ├── lambda_process_upload.py     # Main orchestration Lambda
│   │   ├── deploy_lambda_functions.sh   # Lambda deployment
│   │   └── lambda_execution_role_policy.json
│   ├── setup_infrastructure.sh          # Core infrastructure setup
│   └── ec2_instance_role_policy.json    # EC2 permissions
├── run_transcription/
│   ├── fast_transcribe.py               # Core transcription logic
│   ├── fast_transcribe.sh               # EC2 worker script
│   └── build_ami.sh                     # AMI creation
├── build_ami.sh                         # Main AMI build script
└── README.md                            # Setup instructions
```

## Setup Commands

```bash
# 1. Build AMI with transcription software
./build_ami.sh

# 2. Deploy infrastructure
cd setup && ./setup_infrastructure.sh

# 3. Deploy Lambda function
cd lambda && ./deploy_lambda_functions.sh

# 4. Test with audio file
aws s3 cp audio.mp3 s3://transcription-curevo/transcription_upload/audio.mp3
``` 