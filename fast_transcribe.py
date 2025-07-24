#!/usr/bin/env python3
"""
Fast GPU Transcription - Simple & Elegant
=========================================
Minimal, reliable transcription using the cached AMI setup.
"""

import sys
import time
import json
import subprocess
import os
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# S3 integration
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    log_with_timestamp = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    log_with_timestamp("Warning: boto3 not available - S3 upload disabled")


def log_with_timestamp(message):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def check_nvidia_setup():
    """Verify NVIDIA drivers and CUDA availability."""
    log_with_timestamp("Checking NVIDIA/CUDA setup...")
    
    # Check if nvidia-smi is available
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        log_with_timestamp("NVIDIA drivers detected")
        
        # Extract GPU info from nvidia-smi output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Tesla' in line or 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                gpu_info = ' '.join(line.split()[1:4])  # Extract GPU name
                log_with_timestamp(f"GPU detected: {gpu_info}")
                break
                
    except (subprocess.CalledProcessError, FileNotFoundError):
        log_with_timestamp("Warning: nvidia-smi not available or failed")
        return False
    
    # Check CUDA availability in PyTorch
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        
        log_with_timestamp(f"CUDA available: v{cuda_version}")
        log_with_timestamp(f"GPU devices: {device_count}")
        log_with_timestamp(f"Primary GPU: {device_name}")
        return True
    else:
        log_with_timestamp("Warning: CUDA not available in PyTorch")
        return False


class FastTranscriber:
    """Simple transcriber matching AMI setup exactly."""
    
    def __init__(self):
        self.model_id = "KBLab/kb-whisper-small"
        self.cache_dir = "/opt/transcribe/models"
        self.s3_bucket = "transcription-curevo"
        
        # Initialize S3 client if available
        if S3_AVAILABLE:
            try:
                self.s3_client = boto3.client('s3')
                log_with_timestamp("S3 client initialized")
            except (ClientError, NoCredentialsError) as e:
                log_with_timestamp(f"Warning: S3 initialization failed: {e}")
                self.s3_client = None
        else:
            self.s3_client = None
        
        # Verify GPU setup
        self.cuda_available = check_nvidia_setup()
        
        # Set device with fallback
        if self.cuda_available:
            self.device = "cuda"
            self.torch_dtype = torch.float16
            log_with_timestamp("Using GPU acceleration (CUDA)")
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32
            log_with_timestamp("Falling back to CPU processing")
            
        # Verify cache directory exists
        if not Path(self.cache_dir).exists():
            raise FileNotFoundError(f"Model cache directory not found: {self.cache_dir}")
        
        log_with_timestamp(f"Model cache directory: {self.cache_dir}")
        
    def load_model(self):
        """Load the cached model - exactly like AMI setup."""
        log_with_timestamp("Loading model from cache...")
        log_with_timestamp(f"Device: {self.device} ({self.torch_dtype})")
        
        start_time = time.time()
        
        try:
            # Load model exactly like AMI script
            log_with_timestamp("Loading speech recognition model...")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=True,
                torch_dtype=self.torch_dtype
            )
            
            if self.device == "cuda":
                log_with_timestamp("Moving model to GPU...")
                self.model = self.model.to(self.device)
                
                # Check GPU memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    log_with_timestamp(f"GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
            log_with_timestamp("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            
            # Create pipeline exactly like AMI script
            log_with_timestamp("Creating transcription pipeline...")
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=30,
                batch_size=8 if self.device == "cuda" else 4,  # Smaller batch for CPU
                torch_dtype=self.torch_dtype,
                device=self.device
            )
            
            load_time = time.time() - start_time
            log_with_timestamp(f"Model loaded successfully in {load_time:.2f} seconds")
            return load_time
            
        except Exception as e:
            log_with_timestamp(f"Error loading model: {str(e)}")
            raise
    
    def transcribe(self, audio_file):
        """Transcribe audio file."""
        log_with_timestamp(f"Starting transcription of: {audio_file}")
        
        # Verify audio file exists
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        start_time = time.time()
        
        try:
            result = self.pipeline(audio_file, return_timestamps=True)
            
            duration = time.time() - start_time
            log_with_timestamp(f"Transcription completed in {duration:.2f} seconds")
            
            # Log transcription stats
            text_length = len(result["text"])
            log_with_timestamp(f"Transcribed text length: {text_length} characters")
            
            return result, duration
            
        except Exception as e:
            log_with_timestamp(f"Error during transcription: {str(e)}")
            raise
    
    def upload_to_s3(self, local_file, audio_file):
        """Upload transcription result to S3."""
        if not self.s3_client:
            log_with_timestamp("S3 client not available - skipping upload")
            return None
            
        try:
            # Create S3 key with timestamp and original filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            audio_basename = Path(audio_file).stem
            s3_key = f"results/{timestamp}/transcription_{audio_basename}.txt"
            
            log_with_timestamp(f"Uploading to S3: s3://{self.s3_bucket}/{s3_key}")
            
            # Upload file to S3
            self.s3_client.upload_file(
                local_file, 
                self.s3_bucket, 
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'Metadata': {
                        'audio_file': audio_basename,
                        'timestamp': timestamp,
                        'model': self.model_id,
                        'device': self.device
                    }
                }
            )
            
            s3_url = f"s3://{self.s3_bucket}/{s3_key}"
            log_with_timestamp(f"Successfully uploaded to: {s3_url}")
            return s3_url
            
        except ClientError as e:
            log_with_timestamp(f"Error uploading to S3: {e}")
            return None
        except Exception as e:
            log_with_timestamp(f"Unexpected error during S3 upload: {e}")
            return None
    
    def save_result(self, result, audio_file, load_time, transcribe_time):
        """Save transcription result locally and upload to S3."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"transcription_{timestamp}.txt"
        
        # Get system info
        gpu_info = "N/A"
        if self.cuda_available and torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
            
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== GPU TRANSCRIPTION RESULT ===\n")
                f.write(f"File: {audio_file}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"GPU: {gpu_info}\n")
                f.write(f"Model: {self.model_id}\n")
                f.write(f"Cache: {self.cache_dir}\n")
                f.write(f"Load time: {load_time:.2f}s\n")
                f.write(f"Transcription time: {transcribe_time:.2f}s\n")
                f.write(f"Total time: {load_time + transcribe_time:.2f}s\n")
                f.write(f"Text length: {len(result['text'])} characters\n")
                f.write("\n=== TRANSCRIPTION ===\n")
                f.write(result["text"])
                
                # Add timestamps if available
                if "chunks" in result and result["chunks"]:
                    f.write("\n\n=== TIMESTAMPS ===\n")
                    for chunk in result["chunks"]:
                        timestamp_info = chunk.get("timestamp", (None, None))
                        if timestamp_info[0] is not None:
                            f.write(f"[{timestamp_info[0]:.2f}s - {timestamp_info[1]:.2f}s]: {chunk['text']}\n")
        
            log_with_timestamp(f"Result saved to: {output_file}")
            
            # Upload to S3
            s3_url = self.upload_to_s3(output_file, audio_file)
            
            return output_file, s3_url
            
        except Exception as e:
            log_with_timestamp(f"Error saving result: {str(e)}")
            raise


def main():
    """Main transcription function."""
    if len(sys.argv) != 2:
        log_with_timestamp("Usage: python3 fast_transcribe.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        log_with_timestamp(f"Error: File not found: {audio_file}")
        sys.exit(1)
    
    log_with_timestamp("=== Fast GPU Transcription ===")
    log_with_timestamp(f"Python version: {sys.version.split()[0]}")
    log_with_timestamp(f"PyTorch version: {torch.__version__}")
    
    try:
        # Initialize transcriber
        transcriber = FastTranscriber()
        
        # Load model
        load_time = transcriber.load_model()
        
        # Transcribe
        result, transcribe_time = transcriber.transcribe(audio_file)
        
        # Save result
        output_file, s3_url = transcriber.save_result(result, audio_file, load_time, transcribe_time)
        
        # Summary
        total_time = load_time + transcribe_time
        log_with_timestamp("=== Summary ===")
        log_with_timestamp(f"Total time: {total_time:.2f}s")
        log_with_timestamp(f"Output file: {output_file}")
        if s3_url:
            log_with_timestamp(f"S3 location: {s3_url}")
        log_with_timestamp("=== Transcription Text ===")
        print(result["text"])
        
    except Exception as e:
        log_with_timestamp(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 