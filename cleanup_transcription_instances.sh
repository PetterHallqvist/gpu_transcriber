#!/bin/bash

# Cleanup script to find and terminate all transcription-related EC2 instances
# This script can be run manually to clean up instances left running from previous sessions

echo "Transcription Instance Cleanup"
echo "=============================="

export AWS_DEFAULT_REGION=eu-north-1

# Function to find running instances with transcription security group
find_transcription_instances() {
    echo "Searching for running transcription instances..."
    
    # Find instances with the transcription security group
    RUNNING_INSTANCES=$(aws ec2 describe-instances \
        --filters "Name=instance-state-name,Values=running" \
                  "Name=instance.group-name,Values=transcription-g4dn-sg" \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime,PublicIpAddress]' \
        --output text)
    
    if [ -z "$RUNNING_INSTANCES" ]; then
        echo "No running transcription instances found."
        return 0
    fi
    
    echo "Found running transcription instances:"
    echo "======================================"
    echo "Instance ID       | Type        | Launch Time              | Public IP"
    echo "------------------|-------------|--------------------------|---------------"
    echo "$RUNNING_INSTANCES" | while read instance_id instance_type launch_time public_ip; do
        printf "%-17s | %-11s | %-24s | %-15s\n" "$instance_id" "$instance_type" "$launch_time" "$public_ip"
    done
    echo ""
    
    # Get instance IDs only
    INSTANCE_IDS=$(echo "$RUNNING_INSTANCES" | awk '{print $1}')
    
    return 1
}

# Function to find pending spot requests
find_spot_requests() {
    echo "Searching for active spot requests..."
    
    SPOT_REQUESTS=$(aws ec2 describe-spot-instance-requests \
        --filters "Name=state,Values=open,active" \
        --query 'SpotInstanceRequests[].[SpotInstanceRequestId,State,InstanceId,CreateTime]' \
        --output text)
    
    if [ -z "$SPOT_REQUESTS" ]; then
        echo "No active spot requests found."
        return 0
    fi
    
    echo "Found active spot requests:"
    echo "=========================="
    echo "Spot Request ID                  | State  | Instance ID      | Create Time"
    echo "---------------------------------|--------|------------------|-------------------"
    echo "$SPOT_REQUESTS" | while read spot_id state instance_id create_time; do
        printf "%-32s | %-6s | %-16s | %-18s\n" "$spot_id" "$state" "$instance_id" "$create_time"
    done
    echo ""
    
    # Get spot request IDs only
    SPOT_REQUEST_IDS=$(echo "$SPOT_REQUESTS" | awk '{print $1}')
    
    return 1
}

# Function to cleanup temp files
cleanup_temp_files() {
    echo "Cleaning up temporary files..."
    rm -f /tmp/transcription_instance_id
    rm -f /tmp/transcription_spot_request_id
    rm -f /tmp/quick_transcription_instance_id
    rm -f /tmp/quick_transcription_spot_request_id
    rm -f /tmp/build_ami_instance_id
    rm -f /tmp/build_ami_spot_request_id
    echo "Temporary files cleaned up."
}

# Main cleanup process
echo "Checking for transcription instances and spot requests..."
echo ""

# Clean up temp files first
cleanup_temp_files
echo ""

# Find instances
find_transcription_instances
instances_found=$?

# Find spot requests
find_spot_requests
spots_found=$?

if [ $instances_found -eq 0 ] && [ $spots_found -eq 0 ]; then
    echo "No cleanup needed. All transcription resources are clean."
    exit 0
fi

echo "WARNING: Found active transcription resources that may be incurring costs!"
echo ""

# Ask for confirmation
read -p "Do you want to terminate ALL found instances and cancel spot requests? (y/N): " -n 1 -r
echo ""
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled. Resources left running."
    echo "To manually terminate instances:"
    if [ $instances_found -eq 1 ]; then
        echo "$INSTANCE_IDS" | while read instance_id; do
            echo "  aws ec2 terminate-instances --instance-ids $instance_id"
        done
    fi
    if [ $spots_found -eq 1 ]; then
        echo "$SPOT_REQUEST_IDS" | while read spot_id; do
            echo "  aws ec2 cancel-spot-instance-requests --spot-instance-request-ids $spot_id"
        done
    fi
    exit 1
fi

echo "Proceeding with cleanup..."
echo ""

# Terminate instances
if [ $instances_found -eq 1 ]; then
    echo "Terminating instances..."
    for instance_id in $INSTANCE_IDS; do
        echo "  Terminating: $instance_id"
        aws ec2 terminate-instances --instance-ids "$instance_id" >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "    ✓ Successfully requested termination"
        else
            echo "    ✗ Failed to terminate"
        fi
    done
    echo ""
fi

# Cancel spot requests
if [ $spots_found -eq 1 ]; then
    echo "Cancelling spot requests..."
    for spot_id in $SPOT_REQUEST_IDS; do
        echo "  Cancelling: $spot_id"
        aws ec2 cancel-spot-instance-requests --spot-instance-request-ids "$spot_id" >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "    ✓ Successfully cancelled"
        else
            echo "    ✗ Failed to cancel"
        fi
    done
    echo ""
fi

echo "Cleanup completed!"
echo ""
echo "Note: It may take a few minutes for instances to fully terminate."
echo "You can check the status with:"
echo "  aws ec2 describe-instances --instance-ids [INSTANCE_ID] --query 'Reservations[].Instances[].State.Name'"
echo ""
echo "Remember to monitor your AWS billing to ensure no unexpected charges." 