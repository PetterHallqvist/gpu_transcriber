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

# Import our optimized components
from optimized_loader import OptimizedModelLoader
from direct_transcribe import DirectTranscriber

# AMI Configuration
EXPECTED_AMI_ID = 'ami-09925fd708c360135'  # Expected optimized GPU-enabled AMI for transcription processing

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
    """Ultra-fast transcriber with optimized loading and direct generation."""
    
    def __init__(self):
        self.model_id = "KBLab/kb-whisper-small"
        self.s3_bucket = "transcription-curevo"
        
        # Initialize optimized loader
        self.loader = OptimizedModelLoader(self.model_id)
        
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
        
        # Device info will be set by loader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log_msg(f"Target device: {self.device}")
        
    def load_model(self):
        """Load model using optimized strategies."""
        log_msg("=== Optimized Model Loading ===")
        
        try:
            # Use optimized loader with automatic strategy selection
            self.model, self.processor, load_time = self.loader.load_optimized()
            
            # Initialize direct transcriber
            self.direct_transcriber = DirectTranscriber(self.model, self.processor)
            
            log_msg(f"✓ Model loaded successfully in {load_time:.2f}s")
            return load_time
            
        except Exception as e:
            log_msg(f"ERROR: Optimized model loading failed: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def transcribe(self, audio_file):
        """Transcribe using optimized direct generation."""
        log_msg(f"=== Direct Transcription ===")
        log_msg(f"File: {audio_file}")
        
        # Verify audio file exists
        if not Path(audio_file).exists():
            log_msg(f"ERROR: Audio file not found: {audio_file}")
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Get file size for logging
        file_size = Path(audio_file).stat().st_size
        log_msg(f"Audio file size: {file_size:,} bytes")
        
        try:
            # Use direct transcription for maximum speed
            result, transcribe_time = self.direct_transcriber.transcribe_direct(audio_file)
            
            # Validate result
            if not result or 'text' not in result:
                log_msg("ERROR: Transcription returned invalid result")
                raise RuntimeError("Direct transcription returned invalid result")
            
            text_length = len(result['text'])
            if text_length == 0:
                log_msg("WARNING: Transcription returned empty text")
            else:
                log_msg(f"✓ Transcription generated {text_length} characters")
            
            # Extract real processing time from metadata
            if 'metadata' in result and 'processing_time' in result['metadata']:
                actual_time = result['metadata']['processing_time']
                rtf = result['metadata'].get('real_time_factor', 0)
                log_msg(f"✓ Real-time factor: {rtf:.2f}x")
            
        except Exception as e:
            log_msg(f"ERROR: Direct transcription failed: {str(e)}")
            raise RuntimeError(f"Direct transcription failed: {str(e)}")
        
        log_msg(f"✓ Direct transcription completed in {transcribe_time:.2f}s")
        
        return result, transcribe_time
    
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
        """Save transcription result with optimized structure."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_uuid = str(uuid.uuid4())[:8]
        
        output_file = f"{timestamp}_{file_uuid}.txt"
        json_file = f"{timestamp}_{file_uuid}.json"
        job_id = os.environ.get('JOB_ID', 'unknown')
        total_time = load_time + transcribe_time
        
        # Save simple text format
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== OPTIMIZED TRANSCRIPTION ===\n")
            f.write(f"File: {audio_file}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Text length: {len(result['text'])} characters\n")
            f.write("\n=== TRANSCRIPTION ===\n")
            f.write(result["text"])
        
        # Simplified JSON structure
        json_data = {
            'job_id': job_id,
            'status': 'completed',
            'transcript': {
                'text': result["text"],
                'language': 'sv',
                'confidence': result.get('confidence', 0.95)
            },
            'metadata': {
                'file': audio_file,
                'timestamp': datetime.now().isoformat(),
                'total_time': f"{total_time:.2f}s",
                'text_length': len(result['text'])
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        log_msg(f"Results saved: {output_file}, {json_file}")
        
        # Upload to S3
        s3_txt_url = self.upload_to_s3(output_file, audio_file, 'txt')
        s3_json_url = self.upload_to_s3(json_file, audio_file, 'json')
        
        return output_file, s3_txt_url, s3_json_url


def main():
    """Main optimized transcription function."""
    log_msg("=== Optimized GPU Transcription ===")
    
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
            log_msg("✓ AMI verification: PASSED")
            
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
        # Initialize optimized transcriber
        log_msg("Initializing optimized transcriber...")
        transcriber = FastTranscriber()
        
        # Load model with optimized strategies
        log_msg("Loading model with optimized strategies...")
        load_time = transcriber.load_model()
        
        # Direct transcription
        log_msg("Starting optimized transcription...")
        result, transcribe_time = transcriber.transcribe(audio_file)
        
        # Save result
        log_msg("Saving results...")
        output_file, s3_txt_url, s3_json_url = transcriber.save_result(result, audio_file, load_time, transcribe_time)
        
        # Performance summary
        total_time = load_time + transcribe_time
        log_msg("=== Optimization Results ===")
        log_msg(f"✓ Model loading: {load_time:.2f}s")
        log_msg(f"✓ Transcription: {transcribe_time:.2f}s")
        log_msg(f"✓ Total time: {total_time:.2f}s")
        log_msg(f"✓ Output: {output_file}")
        
        # Extract performance metrics from result metadata
        if 'metadata' in result:
            metadata = result['metadata']
            if 'real_time_factor' in metadata:
                log_msg(f"✓ Real-time factor: {metadata['real_time_factor']:.2f}x")
            if 'audio_duration' in metadata:
                log_msg(f"✓ Audio duration: {metadata['audio_duration']:.1f}s")
        
        if s3_txt_url:
            log_msg(f"✓ S3 Text: {s3_txt_url}")
        if s3_json_url:
            log_msg(f"✓ S3 JSON: {s3_json_url}")
        
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