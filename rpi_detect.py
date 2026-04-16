"""
Deteccao de cedulas no Raspberry Pi 4 com modulo de camera.

Modos de uso:
    python rpi_detect.py --model best.pt                    # Captura unica (tira foto, detecta, salva)
    python rpi_detect.py --model best.pt --live             # Preview ao vivo (precisa de monitor/VNC)
    python rpi_detect.py --model best.pt --loop --interval 5  # Loop: detecta a cada N segundos
    python rpi_detect.py --model best.pt --image foto.jpg   # Detecta em imagem existente
"""

import argparse
import time
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


def open_camera(width=640, height=480):
    """Abre a camera do RPi via OpenCV (V4L2 backend)."""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback sem especificar backend
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERRO: Nao conseguiu abrir a camera.")
        print("Verifique:")
        print("  - Camera habilitada: sudo raspi-config -> Interface -> Camera")
        print("  - Cabo conectado corretamente")
        print("  - Teste com: libcamera-still -o test.jpg")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def detect_frame(model, frame, conf=0.5):
    """Roda deteccao em um frame e retorna resultados estruturados."""
    results = model(frame, conf=conf, verbose=False)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": [round(v, 1) for v in [x1, y1, x2, y2]],
            "center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
        })
    annotated = results[0].plot()
    return detections, annotated


def print_detections(detections, inference_time):
    """Imprime deteccoes no console."""
    print(f"\n--- {len(detections)} cedula(s) detectada(s) [{inference_time:.2f}s] ---")
    for d in detections:
        print(f"  {d['class']:>10s}  conf={d['confidence']:.2f}  "
              f"centro=({d['center'][0]:.0f}, {d['center'][1]:.0f})")
    if not detections:
        print("  Nenhuma cedula encontrada.")


def mode_single(model, conf):
    """Captura uma foto, detecta, salva resultado."""
    print("Capturando foto...")
    cap = open_camera()
    # Descarta primeiros frames (autoexposure)
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERRO: Falha ao capturar frame.")
        return

    t0 = time.time()
    detections, annotated = detect_frame(model, frame, conf)
    dt = time.time() - t0

    print_detections(detections, dt)

    cv2.imwrite("captura_original.jpg", frame)
    cv2.imwrite("captura_detectada.jpg", annotated)
    print(f"\nSalvo: captura_original.jpg, captura_detectada.jpg")

    # Salva JSON com deteccoes (util pro sistema de pick-and-place)
    with open("deteccoes.json", "w") as f:
        json.dump(detections, f, indent=2)
    print(f"Deteccoes salvas em: deteccoes.json")


def mode_loop(model, conf, interval):
    """Loop continuo: detecta a cada N segundos."""
    print(f"Modo loop: detectando a cada {interval}s (Ctrl+C para parar)")
    cap = open_camera()
    # Warmup
    for _ in range(10):
        cap.read()

    count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERRO: Falha ao capturar.")
                break

            t0 = time.time()
            detections, annotated = detect_frame(model, frame, conf)
            dt = time.time() - t0

            count += 1
            print_detections(detections, dt)

            # Salva ultima deteccao
            cv2.imwrite("ultima_deteccao.jpg", annotated)
            with open("deteccoes.json", "w") as f:
                json.dump(detections, f, indent=2)

            time.sleep(max(0, interval - dt))
    except KeyboardInterrupt:
        print(f"\nParado apos {count} capturas.")
    finally:
        cap.release()


def mode_live(model, conf):
    """Preview ao vivo com deteccao (precisa de display)."""
    print("Modo ao vivo (pressione 'q' para sair, 's' para salvar)")
    cap = open_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        detections, annotated = detect_frame(model, frame, conf)
        dt = time.time() - t0

        # FPS no canto
        fps_text = f"{1/dt:.1f} FPS ({dt:.2f}s)"
        cv2.putText(annotated, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Cedulas - RPi4", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("snapshot.jpg", annotated)
            print("Snapshot salvo!")

    cap.release()
    cv2.destroyAllWindows()


def mode_image(model, image_path, conf):
    """Detecta em uma imagem existente."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERRO: Nao conseguiu abrir {image_path}")
        return

    t0 = time.time()
    detections, annotated = detect_frame(model, frame, conf)
    dt = time.time() - t0

    print_detections(detections, dt)

    out_path = f"resultado_{Path(image_path).stem}.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Salvo: {out_path}")

    with open("deteccoes.json", "w") as f:
        json.dump(detections, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteccao de cedulas - Raspberry Pi 4")
    parser.add_argument("--model", type=str, required=True, help="Caminho pro .pt")
    parser.add_argument("--conf", type=float, default=0.5, help="Confianca minima (0-1)")
    parser.add_argument("--live", action="store_true", help="Preview ao vivo")
    parser.add_argument("--loop", action="store_true", help="Loop continuo")
    parser.add_argument("--interval", type=float, default=5, help="Intervalo do loop em segundos")
    parser.add_argument("--image", type=str, help="Detectar em imagem")
    args = parser.parse_args()

    print(f"Carregando modelo: {args.model}")
    model = YOLO(args.model)
    print(f"Modelo carregado. Classes: {list(model.names.values())}")

    if args.image:
        mode_image(model, args.image, args.conf)
    elif args.live:
        mode_live(model, args.conf)
    elif args.loop:
        mode_loop(model, args.conf, args.interval)
    else:
        mode_single(model, args.conf)
