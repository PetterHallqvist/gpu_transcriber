#!/usr/bin/env python3
import sys
import os
import time
import torch
import warnings
import json
import traceback
import subprocess
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore")

def log_with_timestamp(message, level="INFO"):
    """Enhanced logging with timestamps and levels"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def check_gpu_status():
    """Comprehensive GPU diagnostics"""
    log_with_timestamp("🔍 GPU Diagnostics Starting...")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    log_with_timestamp(f"PyTorch CUDA Available: {cuda_available}")
    
    if cuda_available:
        log_with_timestamp(f"CUDA Version: {torch.version.cuda}")
        log_with_timestamp(f"GPU Count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            log_with_timestamp(f"GPU 0 Name: {torch.cuda.get_device_name(0)}")
            log_with_timestamp(f"GPU 0 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            log_with_timestamp(f"nvidia-smi output: {result.stdout.strip()}")
        else:
            log_with_timestamp(f"nvidia-smi failed: {result.stderr}", "ERROR")
    except Exception as e:
        log_with_timestamp(f"nvidia-smi not available: {e}", "WARNING")
    
    # Check for NVIDIA drivers
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            log_with_timestamp("✅ NVIDIA drivers working")
        else:
            log_with_timestamp("❌ NVIDIA drivers not working", "ERROR")
    except Exception as e:
        log_with_timestamp(f"❌ NVIDIA drivers not found: {e}", "ERROR")
    
    return cuda_available

def check_environment():
    """Check the transcription environment setup"""
    log_with_timestamp("🔍 Environment Diagnostics...")
    
    # Check directory structure
    paths_to_check = [
        "/opt/transcribe",
        "/opt/transcribe/venv",
        "/opt/transcribe/models",
        "/opt/transcribe/cache",
        "/opt/transcribe/scripts",
        "/opt/transcribe/cache/cache_info.json"
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            if os.path.isdir(path):
                file_count = len(os.listdir(path)) if os.path.isdir(path) else 0
                log_with_timestamp(f"✅ {path} exists ({file_count} items)")
            else:
                file_size = os.path.getsize(path)
                log_with_timestamp(f"✅ {path} exists ({file_size} bytes)")
        else:
            log_with_timestamp(f"❌ {path} missing", "ERROR")
    
    # Check Python environment
    log_with_timestamp(f"Python version: {sys.version}")
    log_with_timestamp(f"PyTorch version: {torch.__version__}")
    
    # Check current working directory
    log_with_timestamp(f"Current directory: {os.getcwd()}")
    log_with_timestamp(f"Scripts in current dir: {[f for f in os.listdir('.') if f.endswith('.py')]}")

def load_cache_config():
    """Load cache configuration with detailed diagnostics"""
    log_with_timestamp("📋 Loading cache configuration...")
    
    cache_file = "/opt/transcribe/cache/cache_info.json"
    
    try:
        if os.path.exists(cache_file):
            log_with_timestamp(f"✅ Cache file found: {cache_file}")
            with open(cache_file, "r") as f:
                cache_info = json.load(f)
            
            log_with_timestamp("📄 Cache configuration:")
            for key, value in cache_info.items():
                log_with_timestamp(f"  {key}: {value}")
            
            # Verify cached models exist
            cache_dir = cache_info.get("cache_dir", "/opt/transcribe/models")
            if os.path.exists(cache_dir):
                model_files = os.listdir(cache_dir)
                log_with_timestamp(f"✅ Cache directory has {len(model_files)} files")
                if len(model_files) > 0:
                    log_with_timestamp(f"  Sample files: {model_files[:3]}")
            else:
                log_with_timestamp(f"❌ Cache directory missing: {cache_dir}", "ERROR")
            
            return cache_info
        else:
            log_with_timestamp(f"⚠️  Cache file not found: {cache_file}", "WARNING")
            log_with_timestamp("Creating fallback configuration...")
            
    except Exception as e:
        log_with_timestamp(f"❌ Error reading cache file: {e}", "ERROR")
        log_with_timestamp(f"Full traceback: {traceback.format_exc()}")
    
    # Create fallback configuration
    fallback_config = {
        "model_id": "KBLab/kb-whisper-small",
        "cache_dir": "/opt/transcribe/models",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_dtype": "torch.float16" if torch.cuda.is_available() else "torch.float32",
        "kernels_compiled": False,
        "fallback": True
    }
    
    log_with_timestamp("📄 Using fallback configuration:")
    for key, value in fallback_config.items():
        log_with_timestamp(f"  {key}: {value}")
    
    return fallback_config

def main():
    start_time = time.time()
    
    log_with_timestamp("🎙️ Enhanced Transcription Starting...")
    log_with_timestamp("="*50)
    
    if len(sys.argv) != 2:
        log_with_timestamp("Usage: python transcribe_optimized.py <audio_file>", "ERROR")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    log_with_timestamp(f"Audio file: {audio_file}")
    
    # Validate audio file
    if not os.path.exists(audio_file):
        log_with_timestamp(f"Audio file not found: {audio_file}", "ERROR")
        sys.exit(1)
    
    file_size = os.path.getsize(audio_file)
    if file_size == 0:
        log_with_timestamp(f"Audio file is empty: {audio_file}", "ERROR")
        sys.exit(1)
    
    log_with_timestamp(f"Audio file size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    # Run diagnostics
    log_with_timestamp("="*50)
    check_environment()
    log_with_timestamp("="*50)
    gpu_available = check_gpu_status()
    log_with_timestamp("="*50)
    
    # Load configuration
    cache_info = load_cache_config()
    
    # Determine final configuration
    model_id = cache_info["model_id"]
    cache_dir = cache_info["cache_dir"]
    device = "cuda" if gpu_available else "cpu"
    torch_dtype = torch.float16 if gpu_available else torch.float32
    
    log_with_timestamp("="*50)
    log_with_timestamp("🚀 Final Configuration:")
    log_with_timestamp(f"  Model: {model_id}")
    log_with_timestamp(f"  Cache dir: {cache_dir}")
    log_with_timestamp(f"  Device: {device}")
    log_with_timestamp(f"  Data type: {torch_dtype}")
    log_with_timestamp(f"  Using fallback: {cache_info.get('fallback', False)}")
    log_with_timestamp("="*50)
    
    if device == "cpu":
        log_with_timestamp("⚠️  WARNING: Using CPU - transcription will be very slow!", "WARNING")
        log_with_timestamp("⚠️  Expected time: 10-30 minutes for 20min audio", "WARNING")
        log_with_timestamp("⚠️  GPU should be available on g4dn.xlarge instance!", "WARNING")
    
    # Load model efficiently
    load_start = time.time()
    log_with_timestamp("📥 Loading model...")
    
    try:
        # Try cache first if it exists
        if not cache_info.get('fallback', False) and os.path.exists(cache_dir):
            log_with_timestamp("🎯 Attempting to load from cache...")
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    cache_dir=cache_dir,
                    device_map="auto" if device == "cuda" else None,
                    local_files_only=True
                )
                processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
                log_with_timestamp("✅ Model loaded from cache successfully!")
            except Exception as e:
                log_with_timestamp(f"⚠️  Cache loading failed: {e}", "WARNING")
                raise  # Re-raise to trigger download fallback
        else:
            log_with_timestamp("⬇️  Cache not available, downloading model...")
            raise Exception("No cache available")  # Trigger download
            
    except Exception as e:
        log_with_timestamp(f"⬇️  Downloading model (cache failed): {str(e)}")
        log_with_timestamp("⏳ This may take 2-5 minutes for first download...")
        
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                cache_dir=cache_dir,
                device_map="auto" if device == "cuda" else None,
                local_files_only=False
            )
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            log_with_timestamp("✅ Model downloaded successfully!")
        except Exception as download_error:
            log_with_timestamp(f"❌ Model download failed: {download_error}", "ERROR")
            log_with_timestamp(f"Full traceback: {traceback.format_exc()}")
            sys.exit(1)
    
    # Configure for device
    if device == "cuda":
        log_with_timestamp("🚀 Applying GPU optimizations...")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        chunk_length_s = 30
        batch_size = 6
        log_with_timestamp(f"  Chunk length: {chunk_length_s}s")
        log_with_timestamp(f"  Batch size: {batch_size}")
    else:
        chunk_length_s = 30
        batch_size = 1
        log_with_timestamp(f"  Chunk length: {chunk_length_s}s")
        log_with_timestamp(f"  Batch size: {batch_size}")
        log_with_timestamp("⚠️  Using CPU optimizations (slower)")
    
    # Create pipeline
    log_with_timestamp("🔧 Creating transcription pipeline...")
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True,
            generate_kwargs={"language": "swedish", "task": "transcribe"}
        )
        
        load_time = time.time() - load_start
        log_with_timestamp(f"⚡ Model ready in {load_time:.1f}s")
        
    except Exception as e:
        log_with_timestamp(f"❌ Pipeline creation failed: {e}", "ERROR")
        log_with_timestamp(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    # Transcribe
    log_with_timestamp("🎯 Starting transcription...")
    if device == "cpu":
        log_with_timestamp("⏰ CPU transcription estimated time: 10-30 minutes")
        log_with_timestamp("💡 Consider checking GPU setup if this is unexpected")
    
    transcribe_start = time.time()
    
    try:
        log_with_timestamp("🔄 Processing audio...")
        result = pipe(audio_file)
        transcription = result["text"]
        transcribe_time = time.time() - transcribe_start
        total_time = time.time() - start_time
        
        log_with_timestamp(f"✅ Transcription completed!")
        log_with_timestamp(f"⏱️  Transcription time: {transcribe_time:.1f}s")
        log_with_timestamp(f"⏱️  Total time: {total_time:.1f}s")
        
        # Performance analysis
        audio_duration_estimate = file_size / (1024 * 1024) * 60  # Rough estimate
        if audio_duration_estimate > 0:
            realtime_factor = transcribe_time / audio_duration_estimate
            log_with_timestamp(f"📊 Performance: {realtime_factor:.2f}x realtime")
            if realtime_factor > 1.0:
                log_with_timestamp(f"⚠️  Slower than realtime - consider GPU optimization", "WARNING")
        
        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"optimized_result_{timestamp}.txt"
        
        with open(result_filename, "w", encoding="utf-8") as f:
            f.write("=== ENHANCED TRANSCRIPTION RESULT ===\n")
            f.write(f"Audio: {audio_file}\n")
            f.write(f"Audio size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)\n")
            f.write(f"Model: {model_id}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Cache used: {not cache_info.get('fallback', False)}\n")
            f.write(f"Transcription Time: {transcribe_time:.1f}s\n")
            f.write(f"Total Time: {total_time:.1f}s\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            if 'realtime_factor' in locals():
                f.write(f"Realtime Factor: {realtime_factor:.2f}x\n")
            f.write("\n=== TRANSCRIPTION RESULT ===\n")
            f.write(transcription)
        
        log_with_timestamp(f"📄 Result saved to: {result_filename}")
        log_with_timestamp(f"📝 Transcription preview: {transcription[:100]}...")
        
    except Exception as e:
        log_with_timestamp(f"❌ Transcription failed: {e}", "ERROR")
        log_with_timestamp(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 