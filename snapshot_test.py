"""Captura 1 frame em 1080p de cada camera, roda deteccao e salva."""
import cv2
import time
from ultralytics import YOLO

MODEL = "runs/detect/runs/cedulas2/weights/best.pt"
CONF = 0.4
TARGET_W, TARGET_H = 1920, 1080
WARMUP_FRAMES = 10

print("Procurando cameras disponiveis...")
cams = []
for idx in range(4):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        continue

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

    frame = None
    for _ in range(WARMUP_FRAMES):
        ret, f = cap.read()
        if ret:
            frame = f
        time.sleep(0.05)

    if frame is not None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cams.append((idx, w, h, frame))
        print(f"  [{idx}] Camera OK  resolucao={w}x{h}")
    cap.release()

if not cams:
    print("Nenhuma camera encontrada!")
    raise SystemExit(1)

model = YOLO(MODEL)

for idx, w, h, frame in cams:
    raw_path = f"snapshot_cam{idx}.jpg"
    out_path = f"snapshot_cam{idx}_det.jpg"
    cv2.imwrite(raw_path, frame)

    results = model(frame, conf=CONF, verbose=False)
    annotated = results[0].plot()
    cv2.imwrite(out_path, annotated)

    print(f"\nCam {idx} ({w}x{h}): {len(results[0].boxes)} deteccao(oes)")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        print(f"  R${cls_name}: conf={confidence:.2f}  centro=({cx:.0f}, {cy:.0f})")
    print(f"  Raw:      {raw_path}")
    print(f"  Anotada:  {out_path}")
