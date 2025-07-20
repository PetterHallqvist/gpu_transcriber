import os
import torch
import json
import warnings
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np

warnings.filterwarnings("ignore")

print("🧠 Advanced Model Caching Starting...")

# Configuration
model_id = "KBLab/kb-whisper-small"
cache_dir = "/opt/transcribe/models"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Device: {device} | Model: {model_id}")

# Enable optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Download and cache model
print("⬇️ Downloading and caching model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    use_safetensors=True,
    cache_dir=cache_dir,
    device_map="auto" if device == "cuda" else None,
    local_files_only=False
)

print("⬇️ Caching processor...")
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

# Pre-warm model with optimizations
if device == "cuda":
    print("🔥 Pre-warming model with optimizations...")
    model = model.half()
    model.eval()
    
    # Create optimized pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=6,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
        generate_kwargs={"language": "swedish", "task": "transcribe"}
    )
    
    # Test with dummy audio to compile kernels
    print("🛠️ Compiling CUDA kernels...")
    dummy_audio = np.random.random(16000 * 10).astype(np.float32)  # 10 seconds
    
    try:
        # This will trigger kernel compilation
        _ = pipe(dummy_audio)
        kernels_compiled = True
        print("✅ CUDA kernels compiled successfully")
    except Exception as e:
        kernels_compiled = False
        print(f"⚠️ Kernel compilation failed: {e}")
    
    # Clean up GPU memory
    torch.cuda.empty_cache()
else:
    kernels_compiled = False
    print("⚠️ CPU mode - no kernel compilation needed")

# Create cache info
cache_info = {
    "model_id": model_id,
    "cache_dir": cache_dir,
    "device": device,
    "torch_dtype": str(torch_dtype),
    "kernels_compiled": kernels_compiled,
    "timestamp": datetime.now().isoformat()
}

# Save cache info
os.makedirs("/opt/transcribe/cache", exist_ok=True)
with open("/opt/transcribe/cache/cache_info.json", "w") as f:
    json.dump(cache_info, f, indent=2)

print("✅ Advanced caching completed!")
print(f"📍 Models cached in: {cache_dir}")
print(f"📄 Cache info saved to: /opt/transcribe/cache/cache_info.json")
print(f"🚀 Kernels compiled: {kernels_compiled}") 