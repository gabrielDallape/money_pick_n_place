"""
Treina YOLOv8 para detecção de cédulas brasileiras.

Uso:
    python train.py --data datasets/cedulas-9fprk/data.yaml
    python train.py --data datasets/moneytimes/data.yaml --epochs 150 --model yolov8s.pt
"""

import argparse
from ultralytics import YOLO


def train(data_yaml: str, model: str = "yolov8n.pt", epochs: int = 100, imgsz: int = 640, batch: int = 16):
    """Treina o modelo YOLO para detecção de cédulas."""
    yolo = YOLO(model)

    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs",
        name="cedulas",
        patience=20,
        save=True,
        plots=True,
    )

    print(f"\nTreino finalizado!")
    print(f"Melhor modelo salvo em: runs/cedulas/weights/best.pt")
    print(f"Métricas: runs/cedulas/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino YOLOv8 para cédulas brasileiras")
    parser.add_argument("--data", type=str, required=True, help="Caminho para o data.yaml do dataset")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Modelo base (yolov8n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz, args.batch)
