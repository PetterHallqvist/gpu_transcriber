#!/usr/bin/env python3
import os
import tempfile
import json
import subprocess
import glob
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

def transcribe_audio(audio_file_path):
    """Run transcription using the optimized system"""
    try:
        # Clean up old results
        old_results = glob.glob('/opt/transcribe/optimized_result_*.txt')
        for old_file in old_results:
            try:
                os.remove(old_file)
            except:
                pass
        
        # Run transcription
        cmd = [
            '/bin/bash', '-c',
            f'cd /opt/transcribe && source venv/bin/activate && python scripts/transcribe_optimized.py "{audio_file_path}"'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            # Find result file
            result_files = glob.glob('/opt/transcribe/optimized_result_*.txt')
            if result_files:
                result_file = max(result_files, key=os.path.getmtime)
                
                with open(result_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract transcription
                if "=== TRANSCRIPTION RESULT ===" in content:
                    transcription = content.split("=== TRANSCRIPTION RESULT ===")[1].strip()
                else:
                    transcription = content
                
                # Clean up
                try:
                    os.remove(result_file)
                except:
                    pass
                
                return {'success': True, 'transcription': transcription}
            else:
                return {'success': False, 'error': 'No result file generated'}
        else:
            return {'success': False, 'error': f'Transcription failed: {result.stderr}'}
            
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Transcription timeout (10 minutes)'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/health', methods=['GET'])
def health():
    """Simple health check"""
    return {'status': 'ready', 'timestamp': datetime.now().isoformat()}

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe uploaded audio file"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4'}
        file_ext = os.path.splitext(secure_filename(file.filename))[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe
            result = transcribe_audio(temp_file_path)
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'transcription': result['transcription'],
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': result['error']}), 500
                
        finally:
            # Clean up temp file
            try:
                os.remove(temp_file_path)
            except:
                pass
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Transcription API Server on port 8000...")
    app.run(host='0.0.0.0', port=8000, debug=False) 