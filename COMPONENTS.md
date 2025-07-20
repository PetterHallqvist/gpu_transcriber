# 🧩 **Modular Components Guide**

This directory contains modular components extracted from `build_ami.sh` to make the architecture more understandable.

## 📁 **Core Components**

### **1. transcribe_optimized.py** - Main Transcription Engine
**Purpose:** Production-grade transcription script optimized for AMI environment
**Key Features:**
- Reads cache configuration from `/opt/transcribe/cache/cache_info.json`
- Falls back to downloading models if cache missing
- T4 GPU optimizations (chunk_size=30, batch_size=6)
- Cache-first loading with `local_files_only=True`
- Saves results as `optimized_result_TIMESTAMP.txt`

**Usage:** `python transcribe_optimized.py audio.mp3`

---

### **2. api_server.py** - Flask REST API Server  
**Purpose:** HTTP API server that wraps the transcription engine
**Key Features:**
- Runs on port 8000 (this is where "port 8000" comes from!)
- `/health` - Complex health check (the slow one we discussed)
- `/transcribe` - POST endpoint for audio file uploads
- File validation (50MB limit, format checking)
- Calls `transcribe_optimized.py` via subprocess

**Usage:** `python api_server.py` (starts Flask on 0.0.0.0:8000)

---

### **3. advanced_cache.py** - Model Caching System
**Purpose:** Pre-downloads models and compiles CUDA kernels during AMI build
**Key Features:**
- Downloads Swedish Whisper model to `/opt/transcribe/models/`
- Pre-compiles CUDA kernels with dummy audio
- Creates `/opt/transcribe/cache/cache_info.json` metadata
- T4 GPU optimizations setup

**Usage:** `python advanced_cache.py` (run once during AMI build)

---

### **4. transcribe-api.service** - Systemd Service
**Purpose:** Auto-starts the API server on boot
**Key Features:**
- Runs `api_server.py` as systemd service
- Auto-restart on failure
- Sets CUDA environment variables
- Logs to system journal

**Usage:** 
```bash
sudo cp transcribe-api.service /etc/systemd/system/
sudo systemctl enable transcribe-api
sudo systemctl start transcribe-api
```

## 🔄 **How Components Work Together**

```
1. AMI Build Time:
   advanced_cache.py → Downloads models → Creates cache_info.json

2. Instance Boot:
   systemd → transcribe-api.service → api_server.py (port 8000)

3. API Request:
   Client → :8000/transcribe → api_server.py → transcribe_optimized.py → Result
```

## 🚀 **What happens when you call the API:**

```bash
curl -X POST -F 'audio=@file.mp3' http://IP:8000/transcribe
```

1. **api_server.py** receives the request
2. Validates file type and size  
3. Saves uploaded file to `/tmp/transcribe_*/`
4. Calls: `python transcribe_optimized.py /tmp/transcribe_*/file.mp3`
5. **transcribe_optimized.py** runs:
   - Reads `/opt/transcribe/cache/cache_info.json`
   - Loads cached model from `/opt/transcribe/models/`
   - Transcribes audio with T4 optimizations
   - Saves result to `optimized_result_TIMESTAMP.txt`
6. **api_server.py** reads result file and returns JSON response

## 🏥 **About the Health Check Issue**

The health check in `api_server.py` is slow because it:
- Imports torch (5-10 seconds)
- Checks GPU status 
- Validates all file paths

**For faster health checks, we could simplify to:**
```python
@app.route('/health')
def health():
    return {'status': 'ready'}  # If Flask responds, API works!
```

## ✨ **Benefits of Modular Approach**

- **Understandable:** Each component has a clear purpose
- **Testable:** Can run components individually 
- **Maintainable:** Easy to modify specific parts
- **Debuggable:** Can trace issues to specific components

## 🔧 **Local Development**

You can run these components locally for testing:
```bash
# Test transcription (needs GPU + dependencies)
python transcribe_optimized.py audio.mp3

# Test API server (needs transcribe_optimized.py)
python api_server.py

# Test caching (for AMI build)
python advanced_cache.py
``` 