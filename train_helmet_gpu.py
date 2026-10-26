from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "work" / "helmet_dataset" / "data.yaml"
BASE_MODEL = PROJECT_DIR / "yolov8s.pt"
OUTPUT_DIR = PROJECT_DIR / "weights"
RUN_NAME = "helmet_gpu_manual"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")
    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch first.")

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=2,
        workers=0,
        device=0,
        amp=False,
        cache=False,
        patience=10,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
