"""
Detecção de cédulas em tempo real via webcam ou imagem.

Uso:
    python detect.py --model runs/cedulas/weights/best.pt                  # webcam
    python detect.py --model runs/cedulas/weights/best.pt --source foto.jpg  # imagem
    python detect.py --model runs/cedulas/weights/best.pt --source 0         # webcam index
"""

import argparse
import cv2
from ultralytics import YOLO


def detect_realtime(model_path: str, source=0, conf: float = 0.5):
    """Executa detecção em tempo real e retorna bounding boxes."""
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Erro: não foi possível abrir a fonte: {source}")
        return

    print("Detecção iniciada. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)

        annotated = results[0].plot()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            print(f"  {cls_name}: {confidence:.2f} | centro=({cx:.0f}, {cy:.0f})")

        cv2.imshow("Detecção de Cédulas", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_image(model_path: str, image_path: str, conf: float = 0.5):
    """Detecta cédulas em uma imagem e retorna as detecções."""
    model = YOLO(model_path)
    results = model(image_path, conf=conf)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class": model.names[cls_id],
            "confidence": float(box.conf[0]),
            "bbox": [x1, y1, x2, y2],
            "center": [(x1 + x2) / 2, (y1 + y2) / 2],
        })

    print(f"\n{len(detections)} cédula(s) detectada(s):")
    for d in detections:
        print(f"  {d['class']}: {d['confidence']:.2f} | centro={d['center']}")

    results[0].save("resultado.jpg")
    print(f"\nImagem salva em: resultado.jpg")

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecção de cédulas brasileiras")
    parser.add_argument("--model", type=str, required=True, help="Caminho para o modelo treinado (.pt)")
    parser.add_argument("--source", type=str, default="0", help="Fonte: webcam (0) ou caminho da imagem")
    parser.add_argument("--conf", type=float, default=0.5, help="Confiança mínima (0-1)")
    args = parser.parse_args()

    if args.source.isdigit():
        detect_realtime(args.model, int(args.source), args.conf)
    else:
        detect_image(args.model, args.source, args.conf)
