#!/bin/bash

# Simple API Server Launcher
# Target: 90 seconds from start to ready
# Give up after: 3 minutes (if not working, something is wrong)

echo "🚀 Simple API Server Launcher"
echo "============================"
echo "Target: Ready in 90 seconds"
echo "Give up after: 3 minutes"
echo ""

START_TIME=$(date +%s)
export AWS_DEFAULT_REGION=eu-north-1

# Simple logging with elapsed time
log() {
    local elapsed=$(($(date +%s) - START_TIME))
    echo "[${elapsed}s] $1"
}

# Cleanup on any error or exit
cleanup() {
    if [ ! -z "$INSTANCE_ID" ]; then
        log "🧹 Cleaning up instance: $INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null 2>&1
    fi
}
trap cleanup EXIT INT TERM

# Check prerequisites
log "🔍 Checking prerequisites..."
if [ ! -f ami_id.txt ]; then
    log "❌ ERROR: Run ./build_ami.sh first"
    exit 1
fi

AMI_ID=$(cat ami_id.txt)
log "✅ Using AMI: $AMI_ID"

# For now, always launch a new server (skip existing check)
log "🚀 Launching fresh API server..."
INSTANCE_OUTPUT=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "g4dn.xlarge" \
    --key-name "transcription-ec2" \
    --security-groups "transcription-g4dn-sg" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=transcription-api}]' \
    --user-data '#!/bin/bash
exec 1> >(logger -s -t transcribe-api) 2>&1
echo "[$(date)] Starting API server..."
cd /opt/transcribe
source venv/bin/activate
sudo systemctl start transcribe-api
echo "[$(date)] API server startup completed"
' \
    --count 1 \
    --output text \
    --query 'Instances[0].InstanceId' 2>&1)

INSTANCE_ID=$(echo "$INSTANCE_OUTPUT" | head -1)

if [ -z "$INSTANCE_ID" ] || echo "$INSTANCE_ID" | grep -q "error"; then
    log "❌ Failed to launch instance: $INSTANCE_OUTPUT"
    exit 1
fi

log "✅ Instance launched: $INSTANCE_ID"

# Wait for instance to be running
log "⏳ Waiting for instance to boot..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

log "✅ Instance running at: $PUBLIC_IP"

# Wait for API to be ready (90 second target, 180 second timeout)
log "⏳ Waiting for API server to start..."
for i in {1..18}; do
    elapsed=$(($(date +%s) - START_TIME))
    log "Health check $i/18 (${elapsed}s elapsed)..."
    
    if curl -s "http://$PUBLIC_IP:8000/health" >/dev/null 2>&1; then
        log "✅ API server is ready!"
        break
    fi
    
    if [ $elapsed -gt 180 ]; then
        log "❌ TIMEOUT: API not ready after 3 minutes"
        log "🔍 Check logs: ssh -i transcription-ec2.pem ubuntu@$PUBLIC_IP 'sudo journalctl -u transcribe-api'"
        exit 1
    fi
    
    sleep 10
done

# Success!
TOTAL_TIME=$(($(date +%s) - START_TIME))
echo ""
log "🎉 API Server Ready! (${TOTAL_TIME}s total)"
echo "=================================="
echo "🌐 Your API endpoints:"
echo "   Transcribe: http://$PUBLIC_IP:8000/transcribe"
echo "   Health:     http://$PUBLIC_IP:8000/health"
echo ""
echo "📝 Usage example:"
echo "   curl -X POST -F 'audio=@your_file.mp3' http://$PUBLIC_IP:8000/transcribe"
echo ""
echo "💰 Cost: ~$1.20/day"
echo "🛑 Stop: aws ec2 terminate-instances --instance-ids $INSTANCE_ID"

# Clear trap - don't auto-cleanup on success
trap - EXIT
echo "$INSTANCE_ID" > api_instance_id.txt 