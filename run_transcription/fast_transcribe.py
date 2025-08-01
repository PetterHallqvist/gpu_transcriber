#!/opt/transcribe/venv/bin/python3
"""
Ultra-Lean GPU Transcription
============================
AMI-optimized: All dependencies pre-installed, minimal verification
Goal: Fast transcription with zero setup overhead
"""

import sys
import time
import os
import json
import uuid
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# AMI Configuration
EXPECTED_AMI_ID = 'ami-06e624dc2b8c11c9f'  # Expected GPU-enabled AMI for transcription processing

# S3 integration
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


def log_msg(message):
    """Simple logging with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


class FastTranscriber:
    """Ultra-lean transcriber - trusts AMI completely."""
    
    def __init__(self):
        self.model_id = "KBLab/kb-whisper-small"
        self.cache_dir = "/opt/transcribe/models"
        self.s3_bucket = "transcription-curevo"
        
        # Initialize S3 client if available
        if S3_AVAILABLE:
            try:
                self.s3_client = boto3.client('s3')
                log_msg("S3 client initialized successfully")
            except Exception as e:
                log_msg(f"S3 client initialization failed: {e}")
                self.s3_client = None
        else:
            log_msg("S3 not available (boto3 not imported)")
            self.s3_client = None
        
        # Set device - trust AMI setup
        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32
            
        log_msg(f"Device: {self.device}")
        
    def load_model(self):
        """Load model from cache with enhanced error handling."""
        log_msg("Loading model from cache...")
        start_time = time.time()
        
        # Verify cache directory exists
        if not Path(self.cache_dir).exists():
            log_msg(f"ERROR: Model cache directory missing: {self.cache_dir}")
            raise FileNotFoundError(f"Model cache directory not found: {self.cache_dir}")
        
        # Check if model files exist in cache
        model_path = Path(self.cache_dir) / self.model_id.replace("/", "--")
        if not model_path.exists():
            log_msg(f"ERROR: Model not found in cache: {model_path}")
            raise FileNotFoundError(f"Model {self.model_id} not cached at {model_path}")
        
        try:
            # Load model from cache
            log_msg(f"Loading model {self.model_id} from {self.cache_dir}")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=True,
                torch_dtype=self.torch_dtype
            )
            log_msg("Model loaded successfully")
            
        except Exception as e:
            log_msg(f"ERROR: Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
        
        try:
            if self.device == "cuda":
                log_msg("Moving model to CUDA...")
                self.model = self.model.to(self.device)
                log_msg("Model moved to CUDA successfully")
                
        except Exception as e:
            log_msg(f"ERROR: Failed to move model to CUDA: {str(e)}")
            raise RuntimeError(f"CUDA transfer failed: {str(e)}")
        
        try:
            log_msg("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            log_msg("Processor loaded successfully")
            
        except Exception as e:
            log_msg(f"ERROR: Failed to load processor: {str(e)}")
            raise RuntimeError(f"Processor loading failed: {str(e)}")
        
        try:
            log_msg("Creating transcription pipeline...")
            # Create pipeline
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=30,
                batch_size=8 if self.device == "cuda" else 4,
                torch_dtype=self.torch_dtype,
                device=self.device
            )
            log_msg("Pipeline created successfully")
            
        except Exception as e:
            log_msg(f"ERROR: Failed to create pipeline: {str(e)}")
            raise RuntimeError(f"Pipeline creation failed: {str(e)}")
        
        load_time = time.time() - start_time
        log_msg(f"Model loaded successfully in {load_time:.2f}s")
        return load_time
    
    def transcribe(self, audio_file):
        """Transcribe audio file with enhanced error handling."""
        log_msg(f"Transcribing: {audio_file}")
        
        # Verify audio file exists
        if not Path(audio_file).exists():
            log_msg(f"ERROR: Audio file not found: {audio_file}")
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Get file size for logging
        file_size = Path(audio_file).stat().st_size
        log_msg(f"Audio file size: {file_size:,} bytes")
        
        start_time = time.time()
        
        try:
            log_msg("Starting transcription pipeline...")
            result = self.pipeline(audio_file, return_timestamps=True)
            
            # Validate result
            if not result or 'text' not in result:
                log_msg("ERROR: Transcription returned invalid result")
                raise RuntimeError("Transcription pipeline returned invalid result")
            
            text_length = len(result['text'])
            if text_length == 0:
                log_msg("WARNING: Transcription returned empty text")
            else:
                log_msg(f"Transcription generated {text_length} characters")
            
        except Exception as e:
            log_msg(f"ERROR: Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription pipeline failed: {str(e)}")
        
        duration = time.time() - start_time
        log_msg(f"Transcription completed successfully in {duration:.2f}s")
        
        return result, duration
    
    def upload_to_s3(self, local_file, audio_file, file_type='txt'):
        """Upload result to S3 with optimized structure."""
        if not self.s3_client:
            log_msg(f"S3 upload skipped: S3 client not available")
            return None
            
        try:
            job_id = os.environ.get('JOB_ID', 'unknown')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
            
            # Use standardized format: trans_DATE_TIME_UUID
            s3_key = f"transcription_results/{job_id}/trans_{timestamp}_{file_uuid}.{file_type}"
            
            log_msg(f"Uploading to S3: s3://{self.s3_bucket}/{s3_key}")
            
            content_type = 'application/json' if file_type == 'json' else 'text/plain'
            self.s3_client.upload_file(
                local_file, 
                self.s3_bucket, 
                s3_key,
                ExtraArgs={'ContentType': content_type}
            )
            
            log_msg(f"S3 upload successful: s3://{self.s3_bucket}/{s3_key}")
            return f"s3://{self.s3_bucket}/{s3_key}"
            
        except Exception as e:
            log_msg(f"Upload failed: {e}")
            return None
    
    def save_result(self, result, audio_file, load_time, transcribe_time):
        """Save transcription result with enhanced JSON structure."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        
        # Use standardized format: trans_DATE_TIME_UUID
        output_file = f"trans_{timestamp}_{file_uuid}.txt"
        json_file = f"trans_{timestamp}_{file_uuid}.json"
        job_id = os.environ.get('JOB_ID', 'unknown')
        audio_basename = Path(audio_file).stem
        
        # Save text format (compact for performance)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== GPU TRANSCRIPTION RESULT ===\n")
            f.write(f"File: {audio_file}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Model: {self.model_id}\n")
            f.write(f"Load time: {load_time:.2f}s\n")
            f.write(f"Transcription time: {transcribe_time:.2f}s\n")
            f.write(f"Total time: {load_time + transcribe_time:.2f}s\n")
            f.write(f"Text length: {len(result['text'])} characters\n")
            f.write("\n=== TRANSCRIPTION ===\n")
            f.write(result["text"])
        
        # Enhanced JSON structure with S3 locations
        total_time = load_time + transcribe_time
        json_data = {
            'job_id': job_id,
            'status': 'completed',
            'transcript': {
                'text': result["text"],
                'language': 'sv',
                'confidence': result.get('confidence', 0.95),
                'segments': result.get('chunks', [])
            },
            'metadata': {
                'file': audio_file,
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'model': self.model_id,
                'processing': {
                    'load_time': f"{load_time:.2f}s",
                    'transcription_time': f"{transcribe_time:.2f}s", 
                    'total_time': f"{total_time:.2f}s"
                },
                'text_length': len(result['text']),
                'version': '1.0'
            },
            's3_locations': {
                'json_result': f"s3://{self.s3_bucket}/transcription_results/{job_id}/trans_{timestamp}_{file_uuid}.json",
                'text_backup': f"s3://{self.s3_bucket}/transcription_results/{job_id}/trans_{timestamp}_{file_uuid}.txt"
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        log_msg(f"Results saved: {output_file}, {json_file}")
        
        # Upload both files to S3
        s3_txt_url = self.upload_to_s3(output_file, audio_file, 'txt')
        s3_json_url = self.upload_to_s3(json_file, audio_file, 'json')
        
        if s3_txt_url:
            log_msg(f"Text uploaded to: {s3_txt_url}")
        if s3_json_url:
            log_msg(f"JSON uploaded to: {s3_json_url}")
        
        return output_file, s3_txt_url, s3_json_url


def main():
    """Main transcription function."""
    log_msg("=== Fast GPU Transcription ===")
    
    # Log environment information
    try:
        # Try to get AMI ID from metadata service
        import urllib.request
        ami_id = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/ami-id', timeout=2).read().decode()
        log_msg(f"Running on AMI: {ami_id}")
        log_msg(f"Expected AMI: {EXPECTED_AMI_ID}")
        
        if ami_id != EXPECTED_AMI_ID:
            log_msg(f"WARNING: Running on unexpected AMI {ami_id}, expected {EXPECTED_AMI_ID}")
        else:
            log_msg("AMI verification: PASSED")
            
    except Exception as e:
        log_msg(f"AMI ID: Unable to retrieve - {str(e)}")
    
    if len(sys.argv) != 2:
        log_msg("ERROR: Invalid arguments")
        log_msg("Usage: python3 fast_transcribe.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    log_msg(f"Audio file: {audio_file}")
    
    if not Path(audio_file).exists():
        log_msg(f"ERROR: Audio file not found: {audio_file}")
        sys.exit(1)
    
    try:
        # Initialize transcriber
        log_msg("Initializing transcriber...")
        transcriber = FastTranscriber()
        
        # Load model
        log_msg("Loading model...")
        load_time = transcriber.load_model()
        
        # Transcribe
        log_msg("Starting transcription...")
        result, transcribe_time = transcriber.transcribe(audio_file)
        
        # Save result
        log_msg("Saving results...")
        output_file, s3_txt_url, s3_json_url = transcriber.save_result(result, audio_file, load_time, transcribe_time)
        
        # Summary
        total_time = load_time + transcribe_time
        log_msg("=== Transcription Complete ===")
        log_msg(f"Total time: {total_time:.2f}s")
        log_msg(f"Output: {output_file}")
        if s3_txt_url:
            log_msg(f"S3 Text: {s3_txt_url}")
        if s3_json_url:
            log_msg(f"S3 JSON: {s3_json_url}")
        
        # Print transcription
        print("\n=== TRANSCRIPTION ===")
        print(result["text"])
        
    except FileNotFoundError as e:
        log_msg(f"ERROR: File not found - {str(e)}")
        sys.exit(1)
    except RuntimeError as e:
        log_msg(f"ERROR: Runtime error - {str(e)}")
        sys.exit(1)
    except Exception as e:
        log_msg(f"ERROR: Unexpected error - {str(e)}")
        log_msg("This may indicate an AMI configuration issue")
        sys.exit(1)


if __name__ == "__main__":
    main() 