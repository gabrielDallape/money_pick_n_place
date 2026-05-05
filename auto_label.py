"""
Auto-label: captura frames da webcam com pre-anotacao via best.pt.

Uso:
    python auto_label.py
    python auto_label.py --model runs/detect/runs/cedulas2/weights/best.pt --conf 0.3

Controles na janela:
    ESPACO  Captura e salva frame + label YOLO
    q       Sair

Saida:
    datasets/new_captures/
        images/    <- .jpg
        labels/    <- .txt no formato YOLO (pronto pra Roboflow)
        classes.txt

Depois, no Roboflow:
    1. Zipa a pasta datasets/new_captures/
    2. Cria novo projeto ou sobe pra v2 do atual
    3. Upload -> formato "YOLOv8"
    4. Label Assist (com custom weights uploadados) revisa automatico
    5. Corrige os que vieram errados, gera nova Version, re-treina
"""

import argparse
import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

DEFAULT_MODEL = "runs/detect/runs/cedulas2/weights/best.pt"
DEFAULT_OUT = "datasets/new_captures"
CLASSES = ["10", "100", "2", "20", "200", "5", "50"]


def to_yolo_line(cls_id, x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main(model_path, conf, out_dir):
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES))

    print(f"Carregando modelo: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    for _ in range(5):
        cap.read()

    if not cap.isOpened():
        print("Erro: webcam nao acessivel")
        return

    w_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {w_cap}x{h_cap}  |  conf={conf}")
    print("Controles:  ESPACO=capturar  q=sair")

    saved = 0
    last_save_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, conf=conf, verbose=False)
        annotated = results[0].plot()

        n_det = len(results[0].boxes)
        hud1 = f"Capturadas: {saved}  |  detec neste frame: {n_det}"
        hud2 = "[ESPACO] salvar   [q] sair"
        cv2.putText(annotated, hud1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, hud2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if time.time() - last_save_time < 0.8:
            flash = annotated.copy()
            cv2.rectangle(flash, (0, 0), (annotated.shape[1], annotated.shape[0]), (0, 255, 0), 20)
            cv2.putText(flash, "SAVED", (annotated.shape[1] // 2 - 80, annotated.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
            cv2.imshow("Auto-Label", flash)
        else:
            cv2.imshow("Auto-Label", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}"
            img_path = os.path.join(img_dir, f"cap_{ts}.jpg")
            lbl_path = os.path.join(lbl_dir, f"cap_{ts}.txt")

            cv2.imwrite(img_path, frame)

            h, w = frame.shape[:2]
            lines = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                lines.append(to_yolo_line(cls_id, x1, y1, x2, y2, w, h))
            with open(lbl_path, "w") as f:
                f.write("\n".join(lines))

            saved += 1
            last_save_time = time.time()

            det_summary = ", ".join(
                f"R${CLASSES[int(b.cls[0])]}({float(b.conf[0]):.2f})"
                for b in results[0].boxes
            ) or "sem deteccao"
            print(f"[{saved:03d}] cap_{ts}.jpg  ->  {det_summary}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal salvo: {saved} imagens em {out_dir}")
    if saved > 0:
        print(f"Proximo passo: zipa a pasta e sobe no Roboflow (formato YOLOv8).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Captura + auto-label pra criar dataset extra")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.3,
                        help="confianca minima (baixo pra pegar casos duvidosos)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.model, args.conf, args.out)
