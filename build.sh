#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# This is a placeholder command to download your YOLO model.
# You will need to find the specific command or URL to download the model file
# you are using and place it in the correct directory.
# For example:
# wget -P ./models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt