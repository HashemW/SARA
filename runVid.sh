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
# VIDEO_ORDER=(
#     C:/Users/hashe/ai_equestrian/input_videos/belgianmare1.mp4
#     C:/Users/hashe/ai_equestrian/input_videos/horseVid.mp4
#     C:/Users/hashe/ai_equestrian/input_videos/IMG_5157.MOV.mp4
#     C:/Users/hashe/ai_equestrian/input_videos/whiteB1.mp4
# )

# for VIDEO_PATH in "${VIDEO_ORDER[@]}"; do
#     echo "Processing video: $VIDEO_PATH"
#     python C:/Users/hashe/ai_equestrian/src/main.py "$VIDEO_PATH" 0
# done
 
python3 /fs/nexus-scratch/hwahed/ai_equestrian/src/main.py /fs/nexus-scratch/hwahed/ai_equestrian/testVideos/horseVid.mp4 0 0