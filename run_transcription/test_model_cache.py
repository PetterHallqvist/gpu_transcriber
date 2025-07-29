#!/usr/bin/env python3
"""
Test script to verify model caching functionality
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

def test_model_cache():
    model_id = "KBLab/kb-whisper-small"
    cache_dir = "/opt/transcribe/models"
    
    print(f"Testing model cache for: {model_id}")
    print(f"Cache directory: {cache_dir}")
    
    # Check if cache directory exists
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"ERROR: Cache directory does not exist: {cache_dir}")
        return False
    
    print(f"✓ Cache directory exists: {cache_dir}")
    
    # Check expected model path
    expected_model_path = cache_path / model_id.replace("/", "--")
    print(f"Expected model path: {expected_model_path}")
    
    if expected_model_path.exists():
        print(f"✓ Model directory exists: {expected_model_path}")
        print(f"Contents: {list(expected_model_path.iterdir())}")
    else:
        print(f"✗ Model directory missing: {expected_model_path}")
        return False
    
    # Try to load model from cache
    try:
        print("Attempting to load model from cache...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.float16
        )
        print("✓ Model loaded successfully from cache")
        
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True
        )
        print("✓ Processor loaded successfully from cache")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load model from cache: {e}")
        return False

if __name__ == "__main__":
    success = test_model_cache()
    sys.exit(0 if success else 1) 