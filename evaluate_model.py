"""Avaliacao detalhada de um modelo treinado: mAP por classe, matriz de confusao,
PR curves, e galeria de imagens onde o modelo errou (FP / FN) para revisao manual.

Saida em: runs/eval/<name>/
    metrics.json             -> mAP, precision, recall por classe
    confusion_matrix.png     -> matriz visual
    classification_report.txt
    errors/                  -> imagens onde o modelo errou, com bbox sobrepostas

Uso:
    python evaluate_model.py --model runs/detect/v2_*/weights/best.pt
    python evaluate_model.py --model runs/detect/v2/weights/best.pt --split test
    python evaluate_model.py --model best.pt --max-errors 100
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def latest_run_best():
    runs = sorted(glob.glob("runs/detect/v2*/weights/best.pt"))
    return runs[-1] if runs else None


def load_yolo_label(path: Path):
    """Le um .txt YOLO -> lista de (cls_id, cx, cy, w, h) normalizados."""
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


def draw_box(img, box, color, label):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def export_error_galleries(model, data_yaml, split, out_dir, conf, iou_thresh, max_errors, names):
    """Roda inferencia em cada imagem do split e salva imagens com erros."""
    with Path(data_yaml).open("r", encoding="utf-8") as f:
        ymeta = yaml.safe_load(f)
    base = Path(ymeta.get("path", Path(data_yaml).parent))
    img_dir = (base / ymeta.get(split if split != "test" else "test", f"{split}/images")).resolve()
    if not img_dir.exists():
        # YAML usa 'val' como key para o split valid
        if split == "valid":
            img_dir = (base / ymeta["val"]).resolve()
    lbl_dir = img_dir.parent / "labels"

    fp_dir = out_dir / "errors" / "false_positives"
    fn_dir = out_dir / "errors" / "false_negatives"
    cls_dir = out_dir / "errors" / "wrong_class"
    for d in (fp_dir, fn_dir, cls_dir):
        d.mkdir(parents=True, exist_ok=True)

    saved = {"fp": 0, "fn": 0, "cls": 0}
    images = sorted([p for p in img_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"Inferencia em {len(images)} imagens do split '{split}' ...")

    for i, img_path in enumerate(images):
        if all(saved[k] >= max_errors for k in saved):
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_yolo_label(lbl_dir / (img_path.stem + ".txt"))
        gt_boxes = [(g[0], yolo_to_xyxy(g[1:], w, h)) for g in gts]

        res = model.predict(str(img_path), conf=conf, verbose=False)[0]
        preds = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                preds.append((int(b.cls[0]), tuple(int(v) for v in b.xyxy[0].tolist()), float(b.conf[0])))

        # match preds <-> gts por IOU
        matched_gt = set()
        matched_pred = set()
        wrong_class = []
        for pi, (pc, pbox, pconf) in enumerate(preds):
            best_iou, best_gi = 0.0, -1
            for gi, (gc, gbox) in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                v = iou(pbox, gbox)
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= iou_thresh:
                matched_pred.add(pi)
                matched_gt.add(best_gi)
                if gt_boxes[best_gi][0] != pc:
                    wrong_class.append((pi, best_gi))

        false_positives = [i for i in range(len(preds)) if i not in matched_pred]
        false_negatives = [i for i in range(len(gt_boxes)) if i not in matched_gt]

        if not false_positives and not false_negatives and not wrong_class:
            continue

        vis = img.copy()
        for pc, pbox, pconf in preds:
            draw_box(vis, pbox, (0, 200, 255), f"pred {names[pc]} {pconf:.2f}")
        for gc, gbox in gt_boxes:
            draw_box(vis, gbox, (0, 255, 0), f"gt {names[gc]}")

        # decide bucket
        if wrong_class and saved["cls"] < max_errors:
            cv2.imwrite(str(cls_dir / img_path.name), vis)
            saved["cls"] += 1
        elif false_positives and saved["fp"] < max_errors:
            cv2.imwrite(str(fp_dir / img_path.name), vis)
            saved["fp"] += 1
        elif false_negatives and saved["fn"] < max_errors:
            cv2.imwrite(str(fn_dir / img_path.name), vis)
            saved["fn"] += 1

    print(f"Erros exportados:")
    print(f"  False positives (modelo viu nota onde nao tinha): {saved['fp']} em {fp_dir}")
    print(f"  False negatives (modelo perdeu nota real):        {saved['fn']} em {fn_dir}")
    print(f"  Classe errada (achou mas classificou errado):     {saved['cls']} em {cls_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Pesos .pt (default: ultimo runs/detect/v2*/weights/best.pt)")
    parser.add_argument("--data", default="datasets/v2/data.yaml")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--name", default=None,
                        help="Nome da pasta de saida (default: <model_dir_name>_<split>)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IOU para considerar pred~gt match")
    parser.add_argument("--max-errors", type=int, default=50,
                        help="Maximo de imagens salvas por categoria de erro")
    parser.add_argument("--no-errors", action="store_true",
                        help="Pula geracao da galeria de erros (so metricas)")
    args = parser.parse_args()

    model_path = args.model or latest_run_best()
    if model_path is None or not Path(model_path).exists():
        raise SystemExit(f"Modelo nao encontrado. Use --model <path>.")
    model_path = Path(model_path).resolve()
    print(f"Modelo: {model_path}")

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    out_name = args.name or f"{model_path.parent.parent.name}_{args.split}"
    out_dir = Path("runs/eval") / out_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saida: {out_dir.resolve()}")

    # ultralytics val: gera mAP, matriz, plots
    print("Rodando ultralytics val ...")
    metrics = model.val(
        data=args.data,
        split="val" if args.split == "valid" else "test",
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        plots=True,
        save_json=True,
        conf=0.001,
        iou=0.6,
    )

    # extrai metricas relevantes
    with Path(args.data).open("r", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]

    summary = {
        "model": str(model_path),
        "split": args.split,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "per_class": {},
    }
    try:
        per_class_p = metrics.box.p
        per_class_r = metrics.box.r
        per_class_map50 = metrics.box.ap50
        per_class_map = metrics.box.ap
        for i, name in enumerate(names):
            if i < len(per_class_p):
                summary["per_class"][name] = {
                    "precision": float(per_class_p[i]),
                    "recall": float(per_class_r[i]),
                    "mAP50": float(per_class_map50[i]),
                    "mAP50-95": float(per_class_map[i]),
                }
    except Exception as e:
        print(f"[aviso] nao consegui extrair metricas por classe: {e}")

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # report textual
    lines = [
        f"Modelo: {model_path}",
        f"Split: {args.split}",
        "",
        f"mAP50:    {summary['mAP50']:.4f}",
        f"mAP50-95: {summary['mAP50-95']:.4f}",
        f"Precision: {summary['precision_mean']:.4f}",
        f"Recall:    {summary['recall_mean']:.4f}",
        "",
        f"{'Classe':<10} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}",
        "-" * 50,
    ]
    for name, m in summary["per_class"].items():
        lines.append(f"R$ {name:<7} {m['precision']:>8.3f} {m['recall']:>8.3f} "
                     f"{m['mAP50']:>8.3f} {m['mAP50-95']:>10.3f}")

    report = "\n".join(lines)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    if not args.no_errors:
        print("\nGerando galeria de erros para revisao...")
        export_error_galleries(
            model, args.data, args.split, out_dir,
            conf=args.conf, iou_thresh=args.iou,
            max_errors=args.max_errors, names=names,
        )

    print(f"\nResultado completo em: {out_dir.resolve()}")
    print("Inspecione errors/false_positives/ pra ver onde o modelo aluc-ina notas")
    print("Inspecione errors/false_negatives/ pra ver notas que ele perde")
    print("Inspecione errors/wrong_class/ pra ver confusao entre denominacoes")


if __name__ == "__main__":
    main()
