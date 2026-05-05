"""Slice metrics: avalia o modelo por FATIAS do dataset, nao so na media.

Por que isso importa:
- Um mAP de 0.85 pode esconder que o modelo tem 0.95 em fotos bem iluminadas
  e 0.55 em fotos escuras. A media nao te diz onde investir.
- Empresas de CV serias avaliam slice-by-slice pra detectar viases e fraquezas.

Fatias calculadas automaticamente:
- rotation_prefix     : '0_', '90_', '180_', '270_' (rotacao no nome do arquivo)
- brightness          : escuro / medio / claro (media do canal V em HSV)
- blur                : nitido / medio / borrado (variancia do laplaciano)
- gt_count            : 1, 2-3, 4+ notas por imagem
- image_area          : pequena, media, grande
- per_class           : por classe presente (R$2, R$5, ...) — sem agrupar

Saida: runs/slices/<name>/
    slices.json   -> metricas por slice
    slices.csv    -> mesma coisa em CSV
    slices.md     -> tabela markdown bonita pra colar em PR/relatorio

Uso:
    python slice_metrics.py --model runs/detect/v2/weights/best.pt
    python slice_metrics.py --split test --conf 0.25
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


def latest_run_best():
    runs = sorted(glob.glob("runs/detect/v2*/weights/best.pt"))
    return runs[-1] if runs else None


def load_yolo_label(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        out.append((int(parts[0]), float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4])))
    return out


def yolo_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    return (
        int((cx - bw / 2) * w), int((cy - bh / 2) * h),
        int((cx + bw / 2) * w), int((cy + bh / 2) * h),
    )


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def image_attributes(img_path: Path, img: np.ndarray, n_gt: int):
    """Calcula atributos da imagem para slicing."""
    h, w = img.shape[:2]
    attrs = {}

    # rotacao no nome
    name = img_path.name
    if name.startswith("0_"):
        attrs["rotation"] = "0"
    elif name.startswith("90_"):
        attrs["rotation"] = "90"
    elif name.startswith("180_"):
        attrs["rotation"] = "180"
    elif name.startswith("270_"):
        attrs["rotation"] = "270"
    else:
        attrs["rotation"] = "no_prefix"

    # brilho (media do V em HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_mean = float(np.mean(hsv[:, :, 2]))
    if v_mean < 80:
        attrs["brightness"] = "escuro"
    elif v_mean < 160:
        attrs["brightness"] = "medio"
    else:
        attrs["brightness"] = "claro"

    # blur (variancia do laplaciano)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_var < 80:
        attrs["sharpness"] = "borrado"
    elif blur_var < 250:
        attrs["sharpness"] = "medio"
    else:
        attrs["sharpness"] = "nitido"

    # contagem de GT
    if n_gt == 0:
        attrs["gt_count"] = "0"
    elif n_gt == 1:
        attrs["gt_count"] = "1"
    elif n_gt <= 3:
        attrs["gt_count"] = "2-3"
    else:
        attrs["gt_count"] = "4+"

    # tamanho da imagem
    pixels = w * h
    if pixels < 640 * 640:
        attrs["resolution"] = "pequena"
    elif pixels < 1280 * 1280:
        attrs["resolution"] = "media"
    else:
        attrs["resolution"] = "grande"

    return attrs


class SliceAggregator:
    """Acumula TP/FP/FN por slice e calcula precision/recall/F1 no fim."""

    def __init__(self):
        # slice_name -> bucket -> {"tp":, "fp":, "fn":}
        self.data = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_imgs": 0}))

    def record(self, attrs, tp, fp, fn):
        for slice_name, bucket in attrs.items():
            agg = self.data[slice_name][bucket]
            agg["tp"] += tp
            agg["fp"] += fp
            agg["fn"] += fn
            agg["n_imgs"] += 1

    def to_rows(self):
        rows = []
        for slice_name, buckets in self.data.items():
            for bucket, m in buckets.items():
                tp, fp, fn = m["tp"], m["fp"], m["fn"]
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                rows.append({
                    "slice": slice_name,
                    "bucket": bucket,
                    "n_imgs": m["n_imgs"],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f1, 4),
                })
        rows.sort(key=lambda r: (r["slice"], r["bucket"]))
        return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default="datasets/v2/data.yaml")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    model_path = args.model or latest_run_best()
    if model_path is None or not Path(model_path).exists():
        raise SystemExit("Modelo nao encontrado.")
    model_path = Path(model_path).resolve()

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    with Path(args.data).open("r", encoding="utf-8") as f:
        ymeta = yaml.safe_load(f)
    names = ymeta["names"]
    base = Path(ymeta.get("path", Path(args.data).parent))
    split_key = "val" if args.split == "valid" else args.split
    img_dir = (base / ymeta[split_key]).resolve()
    lbl_dir = img_dir.parent / "labels"

    out_name = args.name or f"{model_path.parent.parent.name}_{args.split}"
    out_dir = Path("runs/slices") / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"Avaliando {len(images)} imagens em fatias...")

    agg = SliceAggregator()
    class_agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for i, img_path in enumerate(images):
        if i % 100 == 0 and i > 0:
            print(f"  {i}/{len(images)}")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_yolo_label(lbl_dir / (img_path.stem + ".txt"))
        gt_boxes = [(g[0], yolo_to_xyxy(g[1:], w, h)) for g in gts]

        res = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        preds = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                preds.append((int(b.cls[0]),
                              tuple(int(v) for v in b.xyxy[0].tolist()),
                              float(b.conf[0])))

        # match
        matched_gt = set()
        matched_pred = set()
        for pi, (pc, pbox, _) in enumerate(preds):
            best_iou, best_gi = 0.0, -1
            for gi, (gc, gbox) in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                v = iou(pbox, gbox)
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= args.iou and gt_boxes[best_gi][0] == pc:
                matched_pred.add(pi)
                matched_gt.add(best_gi)

        tp = len(matched_pred)
        fp = len(preds) - tp
        fn = len(gt_boxes) - len(matched_gt)

        attrs = image_attributes(img_path, img, len(gt_boxes))
        agg.record(attrs, tp, fp, fn)

        # per-class agg (com base no tipo de cada GT/pred)
        for gi, (gc, _) in enumerate(gt_boxes):
            if gi in matched_gt:
                class_agg[names[gc]]["tp"] += 1
            else:
                class_agg[names[gc]]["fn"] += 1
        for pi, (pc, _, _) in enumerate(preds):
            if pi not in matched_pred:
                class_agg[names[pc]]["fp"] += 1

    rows = agg.to_rows()

    # per-class
    class_rows = []
    for cls, m in class_agg.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        class_rows.append({
            "slice": "class",
            "bucket": f"R${cls}",
            "n_imgs": "—",
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        })
    class_rows.sort(key=lambda r: r["bucket"])
    rows.extend(class_rows)

    # JSON
    (out_dir / "slices.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # CSV
    with (out_dir / "slices.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown
    md = ["# Slice metrics", "",
          f"Modelo: `{model_path.name}`  ", f"Split: `{args.split}`",
          f"Conf: {args.conf}, IoU: {args.iou}", ""]
    current_slice = None
    for r in rows:
        if r["slice"] != current_slice:
            current_slice = r["slice"]
            md.append("")
            md.append(f"## {current_slice}")
            md.append("")
            md.append("| bucket | n_imgs | TP | FP | FN | P | R | F1 |")
            md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        md.append(f"| {r['bucket']} | {r['n_imgs']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                  f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    (out_dir / "slices.md").write_text("\n".join(md), encoding="utf-8")

    # destaques: pior bucket por slice
    print("\n" + "=" * 60)
    print("PIORES FATIAS (F1 mais baixo, com >=10 imagens):")
    by_slice = defaultdict(list)
    for r in rows:
        if isinstance(r["n_imgs"], int) and r["n_imgs"] >= 10:
            by_slice[r["slice"]].append(r)
    for sl, rs in by_slice.items():
        rs.sort(key=lambda x: x["f1"])
        worst = rs[0]
        best = rs[-1]
        print(f"  {sl:15s}  pior: {worst['bucket']:12s} F1={worst['f1']:.3f}  "
              f"melhor: {best['bucket']:12s} F1={best['f1']:.3f}")

    print(f"\nDetalhes: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
