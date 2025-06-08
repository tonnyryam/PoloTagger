# PoloTagger

### 🐍 Environment Setup (Recommended)

We recommend using [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for easy environment setup and package management.

Create and activate an environment:
```bash
conda create -n waterpolo-ai python=3.10
conda activate waterpolo-ai

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