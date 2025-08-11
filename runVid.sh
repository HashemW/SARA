#!/bin/bash

#SBATCH -N 1                  # Use 1 node
#SBATCH -n 1                  # Run 1 task (process)
#SBATCH --gres=gpu:4         # Request 4 GPUs
#SBATCH -t 24:0:0             # Time limit: 24 hours
#SBATCH -p tron               # Partition
#SBATCH --qos=high            # QoS

#SBATCH --job-name=train      # Job name
#SBATCH --cpus-per-task=16    # CPUs per task
#SBATCH --mem=64G             # Memory per node
# Log paths
#SBATCH -o /fs/nexus-scratch/hwahed/ai_equestrian/logs/%j.out
#SBATCH -e /fs/nexus-scratch/hwahed/ai_equestrian/logs/%j.err
# set -e
VIDEO_ORDER=(
    /fs/nexus-scratch/hwahed/ai_equestrian/testVideos/belgianmare1.mp4
    /fs/nexus-scratch/hwahed/ai_equestrian/testVideos/horseVid.mp4
    /fs/nexus-scratch/hwahed/ai_equestrian/testVideos/IMG_5157.MOV.mp4
    /fs/nexus-scratch/hwahed/ai_equestrian/testVideos/whiteB1.mp4
)

for VIDEO_PATH in "${VIDEO_ORDER[@]}"; do
    echo "Processing video: $VIDEO_PATH"
    python /fs/nexus-scratch/hwahed/ai_equestrian/src/main.py "$VIDEO_PATH" 0 0 
done
 
# python3 /fs/nexus-scratch/hwahed/ai_equestrian/src/main.py /fs/nexus-scratch/hwahed/ai_equestrian/testVideos/IMG_5157.MOV.mp4 0 0