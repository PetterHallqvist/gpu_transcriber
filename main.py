import sys
from datetime import datetime
import time
import os

# Get audio file from command line argument or use default
audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio.mp3"

start_time = time.time()
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting G4DN.XLARGE GPU transcription...")
print(f"Audio file: {audio_file}")

# Check if audio file exists
if not os.path.exists(audio_file):
    print(f"ERROR: Audio file '{audio_file}' not found!")
    sys.exit(1)

try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Libraries imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import libraries: {e}")
    sys.exit(1)

# G4DN.XLARGE specific configuration (NVIDIA T4 GPU)
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking T4 GPU availability...")
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16  # T4 optimized with half precision
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] T4 GPU detected: {gpu_name}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU memory: {gpu_memory:.1f} GB")
else:
    print("WARNING: CUDA not available, falling back to CPU")
    device = "cpu"
    torch_dtype = torch.float32

model_id = "KBLab/kb-whisper-small"

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using T4 GPU optimized processing")
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Device: {device}")
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data type: {torch_dtype}")

# Load model with T4 GPU optimizations
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading model: {model_id}")
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        use_safetensors=True, 
        cache_dir="model_cache",
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model loaded successfully")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    sys.exit(1)

try:
    model.to(device)
    if device == "cuda":
        # T4 GPU specific optimizations
        model = model.half()  # Half precision for T4
        torch.backends.cudnn.benchmark = True  # Optimize for T4
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for T4
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model moved to {device}")
except Exception as e:
    print(f"ERROR: Failed to move model to {device}: {e}")
    sys.exit(1)

# Load processor
try:
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processor loaded")
except Exception as e:
    print(f"ERROR: Failed to load processor: {e}")
    sys.exit(1)

# Create pipeline
try:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating T4 optimized pipeline...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pipeline created successfully")
except Exception as e:
    print(f"ERROR: Failed to create pipeline: {e}")
    sys.exit(1)

# T4 GPU optimized settings (16GB VRAM)
if device == "cuda":
    chunk_length_s = 30   # Optimal for T4 16GB memory
    batch_size = 6        # T4 optimized batch size
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] T4 GPU Mode: chunk_size={chunk_length_s}s, batch_size={batch_size}")
else:
    chunk_length_s = 60   # CPU fallback
    batch_size = 4
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CPU Fallback Mode: chunk_size={chunk_length_s}s, batch_size={batch_size}")

# T4 optimized generation settings
generate_kwargs = {
    "task": "transcribe", 
    "language": "sv",
    "use_cache": True,
    "do_sample": False,
    "temperature": 0.0,
    "no_repeat_ngram_size": 3
}

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting T4 GPU transcription of: {audio_file}")
transcription_start = time.time()

# Clear T4 GPU cache before processing
if device == "cuda":
    torch.cuda.empty_cache()

# Run transcription with T4 optimized settings
try:
    res = pipe(
        audio_file, 
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        generate_kwargs=generate_kwargs,
        return_timestamps=True
    )
    
    transcription_end = time.time()
    total_time = time.time() - start_time
    transcription_time = transcription_end - transcription_start
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Transcription completed successfully!")
    print(f"Transcription time: {transcription_time:.2f} seconds")
    print(f"Total runtime: {total_time:.2f} seconds")
    
    if device == "cuda":
        # Report T4 GPU memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"T4 GPU memory used: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    
    print("=" * 60)
    print("TRANSCRIPTION RESULT:")
    print("=" * 60)
    print(res)
    
except Exception as e:
    print(f"ERROR: Transcription failed: {e}")
    if device == "cuda":
        print(f"T4 GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
        print(f"T4 GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB")
    sys.exit(1)

# Clean up T4 GPU memory
if device == "cuda":
    torch.cuda.empty_cache()

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] T4 GPU transcription completed successfully")