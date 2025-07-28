#!/bin/bash

# Wrapper script for fast transcription
# This script calls the actual transcription script in the run_transcription folder

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <audio_file>"
    echo ""
    echo "This script runs fast GPU transcription using the cached AMI."
    echo "The actual transcription script is located in run_transcription/fast_transcribe.sh"
    exit 1
fi

# Call the actual transcription script
exec ./run_transcription/fast_transcribe.sh "$@" 