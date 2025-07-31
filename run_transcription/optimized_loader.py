#!/opt/transcribe/venv/bin/python3
"""
Optimized Model Loader
======================
Elegant loading strategies: GPU state → Memory mapping → Standard fallback
Zero-overhead model loading with intelligent fallbacks
"""

import torch
import os
import time
import json
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def log_msg(message):
    """Simple logging with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


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
            from transformers import AutoConfig
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
            # Use optimized loading parameters
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=True,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,  # Memory optimization
                device_map="auto" if self.device == "cuda" else None
            )
            
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


def test_loading_strategies():
    """Test all loading strategies and report performance."""
    log_msg("=== Testing Loading Strategies ===")
    
    loader = OptimizedModelLoader()
    
    # Test each strategy
    strategies = [
        ("GPU State", loader.load_from_gpu_state),
        ("Memory Mapping", loader.load_with_memory_mapping),
        ("Standard", loader.load_standard)
    ]
    
    results = {}
    
    for name, method in strategies:
        log_msg(f"Testing {name}...")
        result = method()
        if result:
            _, _, load_time = result
            results[name] = load_time
            log_msg(f"✓ {name}: {load_time:.2f}s")
        else:
            results[name] = None
            log_msg(f"✗ {name}: Failed")
    
    # Report results
    log_msg("=== Performance Summary ===")
    for name, time_taken in results.items():
        if time_taken:
            log_msg(f"{name}: {time_taken:.2f}s")
        else:
            log_msg(f"{name}: Not available")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_loading_strategies()
    else:
        # Test optimized loading
        loader = OptimizedModelLoader()
        try:
            model, processor, load_time = loader.load_optimized()
            log_msg(f"Model loaded successfully in {load_time:.2f}s")
        except Exception as e:
            log_msg(f"ERROR: {e}")
            sys.exit(1)