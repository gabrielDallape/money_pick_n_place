"""Teste em tempo real: deteccao + centro + label + ponto de pega em notas dobradas.

Uso:
    python realtime_test.py
    python realtime_test.py --cam 1 --conf 0.45
    python realtime_test.py --model runs/detect/runs/cedulas2/weights/best.pt
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = "runs/detect/runs/cedulas2/weights/best.pt"

# nota brasileira: 142mm x 65mm -> ~2.18. Abaixo desse limite consideramos dobrada.
FOLDED_ASPECT_THRESHOLD = 1.7

# limites de sanidade para descartar falsos positivos do modelo v1
SANITY_MIN_ASPECT = 1.05   # bbox quase quadrada pequena -> ruido
SANITY_MAX_ASPECT = 4.5    # bbox muito fina -> nao e cedula

COLOR_FLAT = (0, 220, 120)
COLOR_FOLDED = (0, 150, 255)
COLOR_CENTER = (255, 255, 255)
COLOR_PICK = (0, 0, 255)
COLOR_HUD_BG = (25, 25, 25)
COLOR_HUD_FG = (240, 240, 240)


def segment_note(roi_bgr):
    """Retorna mascara binaria do maior objeto (nota) dentro do ROI, ou None."""
    if roi_bgr.size == 0:
        return None
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # se o centro estiver "preto", inverte (assume nota mais brilhante que fundo, ou vice-versa)
    h, w = mask.shape
    if mask[h // 2, w // 2] == 0:
        mask = cv2.bitwise_not(mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.05 * h * w:
        return None

    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean, largest


def analyze_note(frame, x1, y1, x2, y2):
    """Determina se a nota esta dobrada e retorna ponto otimo de pega (em coords do frame)."""
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w_box, h_box = x2 - x1, y2 - y1

    roi = frame[y1:y2, x1:x2]
    seg = segment_note(roi)

    if seg is None:
        bbox_aspect = max(w_box, h_box) / max(min(w_box, h_box), 1)
        is_folded = bbox_aspect < FOLDED_ASPECT_THRESHOLD
        return is_folded, (cx, cy), max(min(w_box, h_box) // 6, 8), float(bbox_aspect)

    mask, contour = seg
    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), _ = rect
    rw, rh = max(rw, 1), max(rh, 1)
    aspect = max(rw, rh) / min(rw, rh)
    is_folded = aspect < FOLDED_ASPECT_THRESHOLD

    # ponto otimo de pega: pixel mais distante de qualquer borda (dentro da nota)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    if not np.isfinite(max_val) or max_val <= 0:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return is_folded, (cx, cy), max(min(x2 - x1, y2 - y1) // 6, 8), aspect

    pick_x = int(x1 + max_loc[0])
    pick_y = int(y1 + max_loc[1])
    safe_max = min(x2 - x1, y2 - y1) // 2
    pick_radius = int(np.clip(max_val * 0.75, 8, safe_max))

    return is_folded, (pick_x, pick_y), pick_radius, float(aspect)


def draw_rounded_box(img, x1, y1, x2, y2, color, thickness=2, r=14):
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_label(img, text, x, y, bg, fg=(255, 255, 255), scale=0.6, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    y_top = y - th - pad * 2
    if y_top < 0:
        y_top = 0
        y = th + pad * 2
    cv2.rectangle(img, (x, y_top), (x + tw + pad * 2, y), bg, -1)
    cv2.putText(img, text, (x + pad, y - pad), font, scale, fg, thickness, cv2.LINE_AA)


def draw_x(img, cx, cy, size, color, thickness=3):
    cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx - size, cy + size), (cx + size, cy - size), color, thickness, cv2.LINE_AA)


def process_frame(frame, model, conf, min_area_frac=0.005, track_state=None, persist_frames=3):
    results = model.track(frame, conf=conf, verbose=False, persist=True, tracker="bytetrack.yaml")
    annotated = frame.copy()
    notes = []
    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * min_area_frac

    seen_ids = set()
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        confidence = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        seen_ids.add(track_id)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1 = max(x1, 0); y1 = max(y1, 0)
        x2 = min(x2, frame.shape[1] - 1); y2 = min(y2, frame.shape[0] - 1)
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            continue
        if bw * bh < min_area:
            continue
        bbox_aspect = max(bw, bh) / max(min(bw, bh), 1)
        if bbox_aspect > SANITY_MAX_ASPECT:
            continue

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        is_folded, pick_pt, pick_r, aspect = analyze_note(frame, x1, y1, x2, y2)

        # filtro de sanidade no aspecto real (apos segmentacao) — descarta ruido tipo chao
        if aspect < SANITY_MIN_ASPECT or aspect > SANITY_MAX_ASPECT:
            continue

        # persistencia temporal: so exibe deteccao que apareceu por N frames seguidos
        if track_state is not None and track_id >= 0:
            track_state[track_id] = track_state.get(track_id, 0) + 1
            if track_state[track_id] < persist_frames:
                continue

        color = COLOR_FOLDED if is_folded else COLOR_FLAT

        # bbox arredondada
        draw_rounded_box(annotated, x1, y1, x2, y2, color, thickness=2, r=14)

        # label da nota (em cima)
        status = "DOBRADA" if is_folded else "OK"
        label_text = f"R$ {cls_name}  {confidence * 100:.0f}%  [{status}]"
        draw_label(annotated, label_text, x1, max(y1 - 4, 24), bg=color, fg=(20, 20, 20), scale=0.6)

        # bolinha branca no centro geometrico
        cv2.circle(annotated, (cx, cy), 7, COLOR_CENTER, -1, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 7, (0, 0, 0), 2, cv2.LINE_AA)

        # se dobrada, marca ponto otimo de pega com X em circulo vermelho
        if is_folded:
            cv2.circle(annotated, pick_pt, pick_r, COLOR_PICK, 2, cv2.LINE_AA)
            draw_x(annotated, pick_pt[0], pick_pt[1], max(pick_r // 2, 8), COLOR_PICK, 3)
            tag_y = pick_pt[1] - pick_r - 6
            tag_x = max(pick_pt[0] - 60, 0)
            draw_label(annotated, "PEGAR AQUI", tag_x, max(tag_y, 22), bg=COLOR_PICK, fg=(255, 255, 255), scale=0.5)

        notes.append({
            "class": cls_name,
            "conf": confidence,
            "center": (cx, cy),
            "folded": is_folded,
            "pick": pick_pt,
            "aspect": aspect,
        })

    # limpa tracks que nao apareceram nesse frame
    if track_state is not None:
        for tid in list(track_state.keys()):
            if tid not in seen_ids:
                del track_state[tid]

    return annotated, notes


def open_camera(cam_idx, width, height):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0] if hasattr(cv2, "CAP_DSHOW") else [0]
    for backend in backends:
        cap = cv2.VideoCapture(cam_idx, backend) if backend else cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
        cap.release()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-area", type=float, default=0.01,
                        help="Area minima da bbox (fracao do frame). Padrao 1 por cento")
    parser.add_argument("--persist", type=int, default=5,
                        help="Quantos frames seguidos antes de exibir uma deteccao")
    args = parser.parse_args()

    print(f"Carregando modelo: {args.model}")
    model = YOLO(args.model)

    cap = open_camera(args.cam, args.width, args.height)
    if cap is None or not cap.isOpened():
        print(f"Nao consegui abrir a camera {args.cam}.")
        return

    win = "Money Pick & Place - teste tempo real"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("Atalhos: [q] sair  [s] salvar frame  [+/-] confianca")
    fps_t0 = time.time()
    fps_count = 0
    fps = 0.0
    conf = args.conf
    track_state = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler frame.")
            break

        annotated, notes = process_frame(
            frame, model, conf,
            min_area_frac=args.min_area,
            track_state=track_state,
            persist_frames=args.persist,
        )

        fps_count += 1
        if fps_count >= 10:
            fps = fps_count / max(time.time() - fps_t0, 1e-6)
            fps_t0 = time.time()
            fps_count = 0

        n_total = len(notes)
        n_folded = sum(1 for n in notes if n["folded"])
        hud = [
            f"FPS: {fps:.1f}",
            f"Conf: {conf:.2f}",
            f"Cedulas: {n_total}  (dobradas: {n_folded})",
        ]
        for i, line in enumerate(hud):
            draw_label(annotated, line, 10, 30 + i * 34, bg=COLOR_HUD_BG, fg=COLOR_HUD_FG, scale=0.6)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = int(time.time())
            path = f"realtime_capture_{ts}.jpg"
            cv2.imwrite(path, annotated)
            print(f"Frame salvo: {path}")
        elif key in (ord("+"), ord("=")):
            conf = min(conf + 0.05, 0.95)
        elif key == ord("-"):
            conf = max(conf - 0.05, 0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
