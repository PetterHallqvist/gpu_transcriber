#!/opt/transcribe/venv/bin/python3
"""
Fast Pipeline Transcription with Optimized Model Cache
======================================================
Elegant, streamlined transcription using pre-cached models
Goal: Ultra-fast transcription with minimal complexity
"""

import sys
import time
import os
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def log_msg(message):
    """Simple logging with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] {message}")

log_msg("=== PYTHON SCRIPT START ===")

# AMI Configuration
EXPECTED_AMI_ID = 'ami-0862833fe45c7055b'  # Optimized GPU AMI with pre-cached models


class FastTranscriber:
    """Elegant transcriber using optimized model cache."""
    
    def __init__(self):
        log_msg("=== FastTranscriber INIT ===")
        self._setup_environment()
        self.load_time = self.load_model()
        self.generation_kwargs = {"language": "sv", "task": "transcribe", "num_beams": 1, "do_sample": False}
        log_msg("✅ FastTranscriber initialized")
    
    def _setup_environment(self):
        """Setup optimized environment variables."""
        cache_dir = "/opt/transcribe/models"
        os.environ.update({
            'TRANSFORMERS_CACHE': cache_dir,
            'HF_HOME': cache_dir,
            'TORCH_HOME': cache_dir,
            'HF_DATASETS_CACHE': "/opt/transcribe/cache"
        })
    
    def load_model(self):
        """Load model from optimized cache."""
        log_msg("=== LOADING CACHED MODEL ===")
        start_time = time.time()
        
        model_id = os.environ.get('MODEL_ID', 'KBLab/kb-whisper-small')
        cache_dir = "/opt/transcribe/models"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        log_msg(f"Model: {model_id}")
        log_msg(f"Cache: {cache_dir}")
        log_msg(f"Device: {device}")
        
        # Load model and processor from cache
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
            local_files_only=True
        )
        
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True
        )
        
        # Create optimized pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device=device,
            chunk_length_s=30,
            batch_size=8
        )
        
        load_time = time.time() - start_time
        log_msg(f"✅ Model loaded from cache in {load_time:.3f}s")
        return load_time
    
    def transcribe(self, audio_file):
        """Transcribe audio using cached model."""
        log_msg(f"=== TRANSCRIPTION ===")
        log_msg(f"File: {audio_file}")
        
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        file_size = Path(audio_file).stat().st_size
        log_msg(f"File size: {file_size:,} bytes")
        
        start_time = time.time()
        result = self.pipe(audio_file, generate_kwargs=self.generation_kwargs)
        transcribe_time = time.time() - start_time
        
        text = result['text'] if isinstance(result, dict) else result
        log_msg(f"✅ Transcribed {len(text)} chars in {transcribe_time:.2f}s")
        
        return {
            'text': text,
            'language': 'sv',
            'confidence': 0.95,
            'metadata': {
                'processing_time': transcribe_time,
                'text_length': len(text),
                'method': 'pipeline_cached'
            }
        }, transcribe_time
    
    def save_result(self, result, audio_file, load_time, transcribe_time):
        """Save transcription results to local files."""
        # Get standardized filename from environment (set by Lambda) to avoid race conditions
        standardized_filename = os.environ.get('STANDARDIZED_FILENAME', 'unknown')
        base_name = standardized_filename.rsplit('.', 1)[0]  # Remove extension
        job_id = os.environ.get('JOB_ID', 'unknown')
        total_time = load_time + transcribe_time
        
        # Save text file with expected naming pattern for shell script compatibility
        txt_file = f"transcription_{base_name}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"=== FAST PIPELINE TRANSCRIPTION ===\n")
            f.write(f"File: {audio_file}\n")
            f.write(f"Standardized Filename: {standardized_filename}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model load: {load_time:.2f}s\n")
            f.write(f"Transcription: {transcribe_time:.2f}s\n")
            f.write(f"Total: {total_time:.2f}s\n")
            f.write(f"Text length: {len(result['text'])} chars\n")
            f.write(f"\n=== TRANSCRIPTION ===\n{result['text']}")
        
        # Save JSON file with consistent naming
        json_file = f"transcription_{base_name}.json"
        json_data = {
            'job_id': job_id,
            'status': 'completed',
            'transcript': {
                'text': result['text'],
                'language': result['language'],
                'confidence': result['confidence']
            },
            'metadata': {
                'file': audio_file,
                'timestamp': datetime.now().isoformat(),
                'model_load_time': f"{load_time:.2f}s",
                'transcription_time': f"{transcribe_time:.2f}s",
                'total_time': f"{total_time:.2f}s",
                **{k: v for k, v in result['metadata'].items() if v is not None}
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        log_msg(f"Saved: {txt_file}, {json_file}")
        return txt_file, None, None


def main():
    """Main transcription function."""
    log_msg("=== Fast Pipeline Transcription ===")
    main_start = time.time()
    
    # Check AMI
    try:
        import urllib.request
        ami_id = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/ami-id', timeout=2).read().decode()
        log_msg(f"✓ AMI: {ami_id}")
    except Exception as e:
        log_msg(f"✗ AMI check failed: {e}")
    
    if len(sys.argv) != 2:
        log_msg("Usage: python3 fast_transcribe.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    if not Path(audio_file).exists():
        log_msg(f"ERROR: File not found: {audio_file}")
        sys.exit(1)
    
    try:
        # Initialize transcriber (loads model automatically)
        transcriber = FastTranscriber()
        
        # Transcribe audio
        result, transcribe_time = transcriber.transcribe(audio_file)
        
        # Save results  
        output_file, _, _ = transcriber.save_result(result, audio_file, transcriber.load_time, transcribe_time)
        
        # Summary
        total_time = time.time() - main_start
        log_msg(f"\n=== SUMMARY ===")
        log_msg(f"✓ Total: {total_time:.2f}s")
        log_msg(f"✓ Output: {output_file}")
        
        print(f"\n=== TRANSCRIPTION ===\n{result['text']}")
        
    except Exception as e:
        total_time = time.time() - main_start
        log_msg(f"ERROR after {total_time:.2f}s: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()