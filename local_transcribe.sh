#!/bin/bash

# Simple local transcription script
# Runs on current instance, no EC2 launching

if [ $# -eq 0 ]; then
    echo "Usage: $0 <audio_file>"
    exit 1
fi

AUDIO_FILE="$1"
echo "[$(date '+%H:%M:%S')] 🎵 Local transcription starting..."
echo "[$(date '+%H:%M:%S')] Audio file: $AUDIO_FILE"

# Check if file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "[$(date '+%H:%M:%S')] ❌ ERROR: File not found: $AUDIO_FILE"
    exit 1
fi

# Check file size
FILE_SIZE=$(stat -c%s "$AUDIO_FILE" 2>/dev/null || stat -f%z "$AUDIO_FILE" 2>/dev/null || echo "unknown")
echo "[$(date '+%H:%M:%S')] File size: $FILE_SIZE bytes"

# Change to transcription directory and activate environment
echo "[$(date '+%H:%M:%S')] 🔧 Setting up environment..."
cd /opt/transcribe || exit 1
source venv/bin/activate || exit 1

# Run transcription
echo "[$(date '+%H:%M:%S')] 🚀 Running transcription (this may take 1-3 minutes)..."
python scripts/transcribe_optimized.py "$AUDIO_FILE"

echo "[$(date '+%H:%M:%S')] ✅ Transcription completed" 