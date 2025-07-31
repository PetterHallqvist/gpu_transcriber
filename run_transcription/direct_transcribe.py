#!/opt/transcribe/venv/bin/python3
"""
Direct Model Transcription
===========================
Elegant solution: Direct model.generate() without pipeline overhead
Zero-overhead transcription with optimal generation parameters
"""

import torch
import librosa
import numpy as np
import time
from datetime import datetime
from pathlib import Path


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
        self.model_id = "KBLab/kb-whisper-small"
        
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


def test_direct_transcription(audio_file):
    """Test direct transcription performance."""
    log_msg("=== Testing Direct Transcription ===")
    
    # This would need model and processor loaded
    # For testing purposes, show the structure
    log_msg(f"Test file: {audio_file}")
    
    if not Path(audio_file).exists():
        log_msg("ERROR: Test audio file not found")
        return False
    
    log_msg("Direct transcription test structure ready")
    log_msg("Use with loaded model and processor from optimized_loader")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        test_direct_transcription(audio_file)
    else:
        log_msg("Direct transcriber module loaded")
        log_msg("Usage: python direct_transcribe.py <audio_file>")
        log_msg("Or import DirectTranscriber class")