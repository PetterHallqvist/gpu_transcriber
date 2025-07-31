#!/opt/transcribe/venv/bin/python3
"""
GPU Memory State Persistence
============================
Pre-loads model into GPU memory during AMI build for instant startup
Elegant solution: Save GPU-loaded state for zero-setup transcription
"""

import torch
import os
import json
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def log_msg(message):
    """Simple logging with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def preload_model_to_gpu():
    """Pre-load model into GPU memory and save optimized state."""
    model_id = "KBLab/kb-whisper-small"
    cache_dir = "/opt/transcribe/models"
    state_dir = "/opt/transcribe/gpu_state"
    
    log_msg("=== GPU Memory State Persistence ===")
    log_msg(f"Model: {model_id}")
    log_msg(f"Cache: {cache_dir}")
    log_msg(f"State: {state_dir}")
    
    # Create state directory
    os.makedirs(state_dir, exist_ok=True)
    
    # Verify CUDA availability
    if not torch.cuda.is_available():
        log_msg("ERROR: CUDA not available - cannot pre-load to GPU")
        return False
    
    log_msg(f"CUDA device: {torch.cuda.get_device_name()}")
    log_msg(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        # Load model and move to GPU
        log_msg("Loading model to GPU...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.float16
        )
        model = model.to('cuda')
        model.eval()  # Set to evaluation mode
        
        log_msg("Model loaded to GPU successfully")
        
        # Load processor
        log_msg("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True
        )
        
        # Save GPU memory state with optimizations
        log_msg("Saving GPU state...")
        gpu_state_file = f"{state_dir}/model_gpu_state.pt"
        
        torch.save({
            'model_state': model.state_dict(),
            'model_config': model.config,
            'device': 'cuda',
            'dtype': str(torch.float16),
            'model_id': model_id,
            'created_at': datetime.now().isoformat()
        }, gpu_state_file)
        
        # Save processor separately
        processor_dir = f"{state_dir}/processor"
        processor.save_pretrained(processor_dir)
        
        # Create info file
        info = {
            'model_id': model_id,
            'gpu_state_file': gpu_state_file,
            'processor_dir': processor_dir,
            'created_at': datetime.now().isoformat(),
            'cuda_device': torch.cuda.get_device_name(),
            'model_size_mb': os.path.getsize(gpu_state_file) / 1024 / 1024,
            'status': 'ready'
        }
        
        with open(f"{state_dir}/gpu_state_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        log_msg(f"GPU state saved: {gpu_state_file}")
        log_msg(f"Processor saved: {processor_dir}")
        log_msg(f"State size: {info['model_size_mb']:.1f}MB")
        log_msg("GPU pre-loading complete!")
        
        return True
        
    except Exception as e:
        log_msg(f"ERROR: GPU pre-loading failed: {e}")
        return False


def verify_gpu_state():
    """Verify that GPU state can be loaded correctly."""
    state_dir = "/opt/transcribe/gpu_state"
    state_file = f"{state_dir}/model_gpu_state.pt"
    
    log_msg("=== Verifying GPU State ===")
    
    if not os.path.exists(state_file):
        log_msg("ERROR: GPU state file not found")
        return False
    
    try:
        # Test loading the state
        log_msg("Testing GPU state loading...")
        state = torch.load(state_file, map_location='cuda')
        
        log_msg(f"✓ Model ID: {state['model_id']}")
        log_msg(f"✓ Device: {state['device']}")
        log_msg(f"✓ Created: {state['created_at']}")
        log_msg("GPU state verification successful")
        
        return True
        
    except Exception as e:
        log_msg(f"ERROR: GPU state verification failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        success = verify_gpu_state()
    else:
        success = preload_model_to_gpu()
        if success:
            success = verify_gpu_state()
    
    sys.exit(0 if success else 1)