# SARA-AI: Equestrian Form Analysis 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-OpenCV-red)

**SARA (System for Automated Rider Analysis)** is an AI-powered computer vision tool designed to analyze horse and rider biomechanics. It uses deep learning to detect gait, grading rider posture (leg, back, and hand positions), and visualizing form corrections on video.

This repository contains the code for SARA-AI, powered by **Modal** (for GPU cloud computing).

Our website is saraai.umd.edu if any equestrian would like to use our model!

## 🚀 Features

* **Multi-Model Pipeline:**
    * **YOLOv8:** Custom-trained detection for horses, riders, and key anatomical points.
    * **Gait Transformer:** A custom Transformer model to classify gaits (Walk, Trot, Canter, Gallop) based on temporal skeletal data.
    * **Geometric Analysis:** Algorithmic grading of rider verticality and joint angles.
