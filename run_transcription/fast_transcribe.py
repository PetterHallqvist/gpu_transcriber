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
import librosa
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoConfig
import boto3

# AMI Configuration
EXPECTED_AMI_ID = 'ami-07b4591c8d4465006'  # Updated optimized GPU-enabled AMI for transcription processing


def log_msg(message):
    """Simple logging with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


class DirectTranscriber:
    """Ultra-fast direct transcription without pipeline overhead."""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device
        self.model_dtype = next(model.parameters()).dtype
        self.model_id = "KBLab/kb-whisper-small"
        
        # Ensure model is in evaluation mode and using consistent dtype
        self.model.eval()
        log_msg(f"Model dtype: {self.model_dtype}")
        
        # Optimal generation parameters for speed
        self.generation_config = {
            'max_length': 448,          # Whisper's max context
            'do_sample': False,         # Deterministic for speed
            'num_beams': 1,            # No beam search for speed
            'language': 'sv',          # Swedish language
            'task': 'transcribe',      # Transcription task
            'use_cache': True,         # Enable KV-cache
            'pad_token_id': self.processor.tokenizer.pad_token_id,
            'eos_token_id': self.processor.tokenizer.eos_token_id
        }
        
        log_msg(f"Direct transcriber initialized on {self.device}")
    
    def load_audio_optimized(self, audio_file):
        """Optimized audio loading with librosa."""
        log_msg(f"Loading audio: {audio_file}")
        start_time = time.time()
        
        try:
            # Load audio at 16kHz (Whisper's expected rate)
            audio, sr = librosa.load(
                audio_file, 
                sr=16000,           # Target sample rate
                mono=True,          # Convert to mono
                dtype=np.float32    # Optimal dtype
            )
            
            load_time = time.time() - start_time
            duration = len(audio) / sr
            
            log_msg(f"✓ Audio loaded: {duration:.1f}s duration, {load_time:.3f}s load time")
            
            return audio, sr
            
        except Exception as e:
            log_msg(f"ERROR: Audio loading failed: {e}")
            raise RuntimeError(f"Audio loading failed: {e}")
    
    def transcribe_direct(self, audio_file):
        """Direct transcription using model.generate() for maximum speed."""
        log_msg(f"=== Direct Transcription ===")
        log_msg(f"File: {audio_file}")
        
        # Verify file exists
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        # Load and preprocess audio
        audio, sr = self.load_audio_optimized(audio_file)
        
        # Process through feature extractor
        log_msg("Processing audio features...")
        feature_start = time.time()
        
        inputs = self.processor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt"
        ).to(self.device)
        
        # Ensure input features match model dtype to prevent type mismatch errors
        inputs["input_features"] = inputs["input_features"].to(dtype=self.model_dtype)
        
        feature_time = time.time() - feature_start
        log_msg(f"✓ Features processed: {feature_time:.3f}s")
        
        # Direct generation without pipeline overhead
        log_msg("Generating transcription...")
        generate_start = time.time()
        
        with torch.no_grad():
            # Use optimized generation parameters
            generated_ids = self.model.generate(
                inputs["input_features"],
                **self.generation_config
            )
        
        generate_time = time.time() - generate_start
        log_msg(f"✓ Generation completed: {generate_time:.3f}s")
        
        # Decode tokens to text
        log_msg("Decoding tokens...")
        decode_start = time.time()
        
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        decode_time = time.time() - decode_start
        log_msg(f"✓ Decoding completed: {decode_time:.3f}s")
        
        # Calculate total time
        total_time = time.time() - start_time
        audio_duration = len(audio) / sr
        rtf = total_time / audio_duration  # Real-time factor
        
        log_msg(f"=== Transcription Summary ===")
        log_msg(f"Audio duration: {audio_duration:.1f}s")
        log_msg(f"Processing time: {total_time:.3f}s")
        log_msg(f"Real-time factor: {rtf:.2f}x")
        log_msg(f"Text length: {len(transcription)} characters")
        
        # Return structured result
        result = {
            'text': transcription,
            'language': 'sv',
            'confidence': 0.95,  # Static confidence for Swedish model
            'metadata': {
                'audio_duration': audio_duration,
                'processing_time': total_time,
                'real_time_factor': rtf,
                'feature_time': feature_time,
                'generation_time': generate_time,
                'decode_time': decode_time
            }
        }
        
        return result, total_time
    
    def transcribe_with_chunking(self, audio_file, chunk_length=30):
        """Direct transcription with chunking for long audio files."""
        log_msg(f"=== Chunked Direct Transcription ===")
        log_msg(f"File: {audio_file}, Chunk length: {chunk_length}s")
        
        # Load full audio
        audio, sr = self.load_audio_optimized(audio_file)
        audio_duration = len(audio) / sr
        
        if audio_duration <= chunk_length:
            # Single chunk transcription
            return self.transcribe_direct(audio_file)
        
        # Split into chunks
        chunk_samples = chunk_length * sr
        chunks = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            chunks.append(chunk)
        
        log_msg(f"Processing {len(chunks)} chunks...")
        
        # Process chunks
        all_text = []
        total_processing_time = 0
        
        for i, chunk in enumerate(chunks):
            log_msg(f"Processing chunk {i+1}/{len(chunks)}...")
            start_time = time.time()
            
            # Process chunk
            inputs = self.processor(
                chunk, 
                sampling_rate=sr, 
                return_tensors="pt"
            ).to(self.device)
            
            # Fix: Ensure input features match model dtype to prevent type mismatch errors
            inputs["input_features"] = inputs["input_features"].to(dtype=self.model_dtype)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    **self.generation_config
                )
            
            chunk_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            all_text.append(chunk_text)
            chunk_time = time.time() - start_time
            total_processing_time += chunk_time
            
            log_msg(f"✓ Chunk {i+1}: {chunk_time:.3f}s")
        
        # Combine results
        full_text = " ".join(all_text)
        rtf = total_processing_time / audio_duration
        
        log_msg(f"=== Chunked Transcription Summary ===")
        log_msg(f"Audio duration: {audio_duration:.1f}s")
        log_msg(f"Total processing: {total_processing_time:.3f}s")
        log_msg(f"Real-time factor: {rtf:.2f}x")
        log_msg(f"Final text length: {len(full_text)} characters")
        
        result = {
            'text': full_text,
            'language': 'sv',
            'confidence': 0.95,
            'metadata': {
                'audio_duration': audio_duration,
                'processing_time': total_processing_time,
                'real_time_factor': rtf,
                'chunks_processed': len(chunks),
                'chunk_length': chunk_length
            }
        }
        
        return result, total_processing_time


class OptimizedModelLoader:
    """Elegant model loader with optimized strategies."""
    
    def __init__(self, model_id="KBLab/kb-whisper-small"):
        self.model_id = model_id
        self.cache_dir = "/opt/transcribe/models"
        self.state_dir = "/opt/transcribe/gpu_state"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
    def load_from_gpu_state(self):
        """Strategy 1: Load from pre-saved GPU state (fastest)."""
        state_file = f"{self.state_dir}/model_gpu_state.pt"
        processor_dir = f"{self.state_dir}/processor"
        
        if not os.path.exists(state_file) or not os.path.exists(processor_dir):
            log_msg("GPU state not available - falling back to next strategy")
            return None
            
        log_msg("Loading from GPU state...")
        start_time = time.time()
        
        try:
            # Load saved GPU state
            state = torch.load(state_file, map_location=self.device)
            
            # Reconstruct model from config and state
            config = AutoConfig.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            
            model = AutoModelForSpeechSeq2Seq.from_config(config)
            model.load_state_dict(state['model_state'])
            model = model.to(self.device)
            model.eval()
            
            # Load processor
            processor = AutoProcessor.from_pretrained(processor_dir)
            
            load_time = time.time() - start_time
            log_msg(f"✓ GPU state loading: {load_time:.2f}s")
            
            return model, processor, load_time
            
        except Exception as e:
            log_msg(f"GPU state loading failed: {e}")
            return None
    
    def load_with_memory_mapping(self):
        """Strategy 2: Memory-mapped loading with optimizations."""
        log_msg("Loading with memory mapping...")
        start_time = time.time()
        
        try:
            # Use optimized loading parameters without accelerate dependency
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=True,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True  # Memory optimization without device_map
            )
            
            # Move to device manually to avoid accelerate dependency
            if self.device == "cuda":
                model = model.to(self.device)
            
            model.eval()
            
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            
            load_time = time.time() - start_time
            log_msg(f"✓ Memory-mapped loading: {load_time:.2f}s")
            
            return model, processor, load_time
            
        except Exception as e:
            log_msg(f"Memory-mapped loading failed: {e}")
            return None
    
    def load_standard(self):
        """Strategy 3: Standard loading (fallback)."""
        log_msg("Loading with standard method...")
        start_time = time.time()
        
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=True,
                torch_dtype=self.torch_dtype
            )
            
            if self.device == "cuda":
                model = model.to(self.device)
            
            model.eval()
            
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            
            load_time = time.time() - start_time
            log_msg(f"✓ Standard loading: {load_time:.2f}s")
            
            return model, processor, load_time
            
        except Exception as e:
            log_msg(f"Standard loading failed: {e}")
            return None
    
    def load_optimized(self):
        """Main method: Try strategies in order of speed."""
        log_msg(f"=== Optimized Model Loading ===")
        log_msg(f"Model: {self.model_id}")
        log_msg(f"Device: {self.device}")
        
        # Strategy 1: GPU state (fastest)
        result = self.load_from_gpu_state()
        if result:
            model, processor, load_time = result
            log_msg(f"SUCCESS: GPU state loading ({load_time:.2f}s)")
            return model, processor, load_time
        
        # Strategy 2: Memory mapping (fast)
        result = self.load_with_memory_mapping()
        if result:
            model, processor, load_time = result
            log_msg(f"SUCCESS: Memory-mapped loading ({load_time:.2f}s)")
            return model, processor, load_time
        
        # Strategy 3: Standard (reliable)
        result = self.load_standard()
        if result:
            model, processor, load_time = result
            log_msg(f"SUCCESS: Standard loading ({load_time:.2f}s)")
            return model, processor, load_time
        
        # All strategies failed
        raise RuntimeError("All loading strategies failed")


class FastTranscriber:
    """Ultra-fast transcriber with optimized loading and direct generation."""
    
    def __init__(self):
        self.model_id = "KBLab/kb-whisper-small"
        self.s3_bucket = "transcription-curevo"
        
        # Initialize optimized loader
        self.loader = OptimizedModelLoader(self.model_id)
        
        # Initialize S3 client if available
        try:
            self.s3_client = boto3.client('s3')
            log_msg("S3 client initialized successfully")
        except Exception as e:
            log_msg(f"S3 client initialization failed: {e}")
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
    
    def transcribe(self, audio_file, chunk_threshold=30):
        """Intelligent transcription with automatic chunking for long files."""
        log_msg(f"=== Intelligent Transcription ===")
        log_msg(f"File: {audio_file}")
        
        # Verify audio file exists
        if not Path(audio_file).exists():
            log_msg(f"ERROR: Audio file not found: {audio_file}")
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Get file size for logging
        file_size = Path(audio_file).stat().st_size
        log_msg(f"Audio file size: {file_size:,} bytes")
        
        try:
            # Check audio duration to decide transcription strategy
            import librosa
            audio_info, sr = librosa.load(audio_file, sr=16000, mono=True, dtype=np.float32)
            duration = len(audio_info) / sr
            log_msg(f"Audio duration: {duration:.1f}s")
            
            if duration <= chunk_threshold:
                # Use direct transcription for short files
                log_msg("Using direct transcription (short audio)")
                result, transcribe_time = self.direct_transcriber.transcribe_direct(audio_file)
            else:
                # Use chunking for long files
                log_msg(f"Using chunked transcription (long audio > {chunk_threshold}s)")
                result, transcribe_time = self.direct_transcriber.transcribe_with_chunking(audio_file)
            
            # Validate result
            if not result or 'text' not in result:
                log_msg("ERROR: Transcription returned invalid result")
                raise RuntimeError("Transcription returned invalid result")
            
            text_length = len(result['text'])
            if text_length == 0:
                log_msg("WARNING: Transcription returned empty text")
            else:
                log_msg(f"✓ Transcription generated {text_length} characters")
            
            # Extract real processing time from metadata
            if 'metadata' in result and 'processing_time' in result['metadata']:
                rtf = result['metadata'].get('real_time_factor', 0)
                log_msg(f"✓ Real-time factor: {rtf:.2f}x")
            
        except Exception as e:
            log_msg(f"ERROR: Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")
        
        log_msg(f"✓ Transcription completed in {transcribe_time:.2f}s")
        
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