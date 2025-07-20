#!/bin/bash

# Launch Production Transcription API Server
# Simple always-on EC2 instance serving transcription requests
# Run once, use forever

echo "Launching Production Transcription API"
echo "====================================="

export AWS_DEFAULT_REGION=eu-north-1

# Cleanup function for graceful shutdown
cleanup_on_error() {
    echo ""
    echo "Cleaning up on error..."
    if [ ! -z "$INSTANCE_ID" ]; then
        echo "Terminating failed instance: $INSTANCE_ID"
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null 2>&1
    fi
}

# Set up signal traps
trap cleanup_on_error EXIT INT TERM

# Check AMI exists
if [ ! -f ami_id.txt ]; then
    echo "ERROR: AMI not found!"
    echo "Please run: ./build_ami.sh first"
    exit 1
fi

AMI_ID=$(cat ami_id.txt)
echo "Using AMI: $AMI_ID"

# Check if API server already running
echo "Checking for existing API server..."
EXISTING_API=$(aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
              "Name=tag:Name,Values=transcription-api" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text 2>/dev/null)

if [ "$EXISTING_API" != "None" ] && [ ! -z "$EXISTING_API" ] && [ "$EXISTING_API" != "null" ]; then
    echo "Found existing API server: $EXISTING_API"
    
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $EXISTING_API \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    echo ""
    echo "🚀 API Server Already Running!"
    echo "=============================="
    echo "Instance ID: $EXISTING_API"
    echo "Public IP: $PUBLIC_IP"
    echo "API URL: http://$PUBLIC_IP:8000/transcribe"
    echo "Health Check: http://$PUBLIC_IP:8000/health"
    echo ""
    echo "Test with:"
    echo "curl -X POST -F 'audio=@your_file.mp3' http://$PUBLIC_IP:8000/transcribe"
    
    # Save instance ID for future reference
    echo "$EXISTING_API" > api_instance_id.txt
    
    # Clear trap since we're not creating a new instance
    trap - EXIT
    exit 0
fi

echo "No existing API server found. Launching new instance..."

# Try multiple zones for best availability
ZONES=("eu-north-1a" "eu-north-1b" "eu-north-1c")
INSTANCE_ID=""

for zone in "${ZONES[@]}"; do
    echo "Trying to launch API server in zone: $zone"
    
    INSTANCE_OUTPUT=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "g4dn.xlarge" \
        --key-name "transcription-ec2" \
        --security-groups "transcription-g4dn-sg" \
        --placement "AvailabilityZone=$zone" \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=transcription-api},{Key=Purpose,Value=api-server}]' \
        --user-data "#!/bin/bash
cd /opt/transcribe
source venv/bin/activate
sudo systemctl start transcribe-api
echo 'API server startup completed' >> /var/log/transcribe-startup.log
" \
        --block-device-mappings '[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": 30,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": true
                }
            }
        ]' \
        --count 1 \
        --output text \
        --query 'Instances[0].InstanceId' 2>&1)
    
    INSTANCE_ID=$(echo "$INSTANCE_OUTPUT" | grep -v "^An error occurred" | head -1)
    
    if [ ! -z "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
        echo "SUCCESS: API server launched in zone: $zone"
        echo "Instance ID: $INSTANCE_ID"
        break
    else
        echo "FAILED: Could not launch in zone $zone"
        if echo "$INSTANCE_OUTPUT" | grep -q "An error occurred"; then
            echo "Error: $INSTANCE_OUTPUT"
        fi
        echo "Trying next zone..."
    fi
done

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
    echo "ERROR: Failed to launch API server in any zone"
    echo "This might be due to:"
    echo "  - G and VT Instance quota exceeded (g4dn.xlarge uses this quota)"
    echo "  - High demand in all zones"
    echo "  - Try again in a few minutes"
    exit 1
fi

# Save instance ID
echo "$INSTANCE_ID" > api_instance_id.txt

echo "Waiting for API server to start..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "API server booting up... waiting for Flask to start..."

# Wait for API to be ready (Flask needs time to load model)
echo "Waiting for API endpoint to respond..."
API_READY=false
for i in {1..60}; do
    if curl -s "http://$PUBLIC_IP:8000/health" >/dev/null 2>&1; then
        API_READY=true
        echo "API is ready!"
        break
    fi
    echo "Attempt $i/60: API starting up..."
    sleep 10
done

if [ "$API_READY" = false ]; then
    echo "WARNING: API health check failed, but instance is running"
    echo "The model might still be loading. Give it a few more minutes."
fi

# Clear trap since everything succeeded
trap - EXIT

echo ""
echo "🚀 API Server Successfully Launched!"
echo "===================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "API URL: http://$PUBLIC_IP:8000/transcribe"
echo "Health Check: http://$PUBLIC_IP:8000/health"
echo ""
echo "Server will run 24/7 until manually stopped"
echo ""
echo "💡 Usage Examples:"
echo "Test health: curl http://$PUBLIC_IP:8000/health"
echo "Transcribe:  curl -X POST -F 'audio=@your_file.mp3' http://$PUBLIC_IP:8000/transcribe"
echo ""
echo "📊 Management:"
echo "Check status: aws ec2 describe-instances --instance-ids $INSTANCE_ID"
echo "Stop server:  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo "Cost: ~\$1.20/day (g4dn.xlarge on-demand pricing)"
echo ""
echo "✅ Setup complete! API server is ready for production use." 