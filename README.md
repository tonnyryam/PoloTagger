# PoloTagger 🎯

## Overview

PoloTagger is an end-to-end video analysis pipeline designed to automatically detect key water polo events such as possessions, shots, goals, man-up/man-down situations, and player cap numbers using deep learning models.

---

## 🐍 Environment Setup

We recommend using **[Anaconda](https://www.anaconda.com/)** or **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)**:

```bash
conda create -n waterpolo-ai python=3.10
conda activate waterpolo-ai

pip install torch torchvision opencv-python pandas scikit-learn moviepy
pip install ultralytics  # For YOLOv8 cap detection

Libraries/Frameworks

Core: PyTorch, OpenCV
CV Models: TorchVision or MMAction2 (for video classification), YOLOv8 (for player detection)
Experiment Tracking: wandb
Data Handling: NumPy, ffmpeg
Configs: Hydra or simple YAML

Initial Features

1. cap numbers in field by team
2. possession
3. goals
4. shots
5. man-up/down

srun -p gpu --gres=gpu:1 --time 1-00:15:00 --pty bash -l