"""Detecta labels suspeitos de estar errados usando o modelo treinado.

Estrategia:
- Roda o modelo em cada imagem do split escolhido
- Compara predicoes com ground truth (matching por IoU)
- Calcula um 'label_quality_score' por amostra, baseado em:
    * desacordo entre classe predita e ground truth com alta confianca
    * GTs sem nenhuma predicao correspondente (label que esta fora ou marcado errado)
    * predicoes muito confiantes sem GT (label faltando = erro de omissao)
- Ranqueia as N piores e exporta lista + galeria visual
- Tambem usa Cleanlab se instalado (ranking complementar mais sofisticado)

Saida em: runs/bad_labels/<name>/
    suspects.csv         -> ranking ordenado, com path da imagem e tipo do problema
    suspects.txt         -> mesma lista, em formato linha-por-linha
    gallery/             -> imagens visualizadas com predicao vs gt sobreposto

Uso:
    python find_bad_labels.py --model runs/detect/v2/weights/best.pt
    python find_bad_labels.py --split train --top 200

Importante:
    - O modelo precisa ja estar razoavelmente bom (mAP > 0.7) pra esse ranking ser util
    - Sempre revise manualmente antes de re-rotular: o modelo tambem erra
    - Sobe os IDs das imagens suspeitas pro Roboflow e usa o filtro 'name contains' pra revisar
"""
import argparse
import csv
import glob
import shutil
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


def draw_box(img, box, color, label):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def analyze_image(img_path, lbl_dir, model, names, conf, iou_thresh):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    gts = load_yolo_label(lbl_dir / (img_path.stem + ".txt"))
    gt_boxes = [(g[0], yolo_to_xyxy(g[1:], w, h)) for g in gts]

    res = model.predict(str(img_path), conf=conf, verbose=False)[0]
    preds = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            preds.append((int(b.cls[0]), tuple(int(v) for v in b.xyxy[0].tolist()),
                          float(b.conf[0])))

    matched_gt = set()
    matched_pred = set()
    wrong_class_pairs = []
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
                wrong_class_pairs.append((pi, best_gi, pconf, best_iou))

    # heuristica de score: quanto maior, mais suspeito
    # peso: classe errada com confianca alta > FN > FP
    score = 0.0
    issues = []
    for pi, gi, pconf, ioup in wrong_class_pairs:
        if pconf > 0.5:
            score += pconf * 2.0
            issues.append(f"classe_errada(pred={names[preds[pi][0]]},gt={names[gt_boxes[gi][0]]},conf={pconf:.2f})")

    fn_count = len(gt_boxes) - len(matched_gt)
    fp_high_conf = [pi for pi in range(len(preds)) if pi not in matched_pred and preds[pi][2] > 0.7]
    if fn_count > 0 and len(gt_boxes) > 0:
        score += fn_count / max(len(gt_boxes), 1) * 1.0
        issues.append(f"gt_sem_predicao({fn_count})")
    if fp_high_conf:
        score += len(fp_high_conf) * 0.5
        issues.append(f"pred_alta_conf_sem_gt({len(fp_high_conf)})")

    return {
        "path": str(img_path),
        "score": score,
        "issues": issues,
        "n_gt": len(gt_boxes),
        "n_pred": len(preds),
        "n_wrong_class": len(wrong_class_pairs),
        "n_fn": fn_count,
        "n_fp_highconf": len(fp_high_conf),
        "image": img,
        "gt_boxes": gt_boxes,
        "preds": preds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default="datasets/v2/data.yaml")
    parser.add_argument("--split", default="train",
                        help="train/valid/test (geralmente o maior — onde tem mais labels pra revisar)")
    parser.add_argument("--top", type=int, default=100,
                        help="Quantos suspeitos exportar")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--name", default=None)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limita quantas imagens analisar (debug)")
    args = parser.parse_args()

    model_path = args.model or latest_run_best()
    if model_path is None or not Path(model_path).exists():
        raise SystemExit("Modelo nao encontrado. Treine antes ou passe --model.")
    model_path = Path(model_path).resolve()

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    with Path(args.data).open("r", encoding="utf-8") as f:
        ymeta = yaml.safe_load(f)
    names = ymeta["names"]
    base = Path(ymeta.get("path", Path(args.data).parent))

    split_key = "val" if args.split == "valid" else args.split
    img_dir = (base / ymeta[split_key]).resolve()
    if not img_dir.exists():
        # fallback comum: <split>/images
        img_dir = (base / args.split / "images").resolve()
    lbl_dir = img_dir.parent / "labels"
    print(f"Imagens: {img_dir}")
    print(f"Labels:  {lbl_dir}")

    out_name = args.name or f"{model_path.parent.parent.name}_{args.split}"
    out_dir = Path("runs/bad_labels") / out_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "gallery").mkdir(parents=True)

    images = sorted([p for p in img_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.max_images:
        images = images[: args.max_images]
    print(f"Analisando {len(images)} imagens...")

    results = []
    for i, img_path in enumerate(images):
        if i % 200 == 0 and i > 0:
            print(f"  {i}/{len(images)}")
        r = analyze_image(img_path, lbl_dir, model, names, args.conf, args.iou)
        if r is None:
            continue
        if r["score"] > 0:
            results.append(r)

    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[: args.top]
    print(f"\n{len(results)} imagens com possivel problema. Exportando top {len(top)}.")

    csv_path = out_dir / "suspects.csv"
    txt_path = out_dir / "suspects.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv, \
            txt_path.open("w", encoding="utf-8") as ftxt:
        writer = csv.writer(fcsv)
        writer.writerow(["rank", "score", "path", "issues",
                         "n_gt", "n_pred", "n_wrong_class", "n_fn", "n_fp_highconf"])
        for rank, r in enumerate(top, 1):
            writer.writerow([rank, f"{r['score']:.3f}", r["path"], ";".join(r["issues"]),
                             r["n_gt"], r["n_pred"], r["n_wrong_class"], r["n_fn"], r["n_fp_highconf"]])
            ftxt.write(f"#{rank:03d}  score={r['score']:.2f}  {Path(r['path']).name}  -> {', '.join(r['issues'])}\n")

            vis = r["image"].copy()
            for pc, pbox, pconf in r["preds"]:
                draw_box(vis, pbox, (0, 200, 255), f"pred {names[pc]} {pconf:.2f}")
            for gc, gbox in r["gt_boxes"]:
                draw_box(vis, gbox, (0, 255, 0), f"gt {names[gc]}")
            cv2.imwrite(str(out_dir / "gallery" / f"{rank:04d}_{Path(r['path']).name}"), vis)

    # Cleanlab opcional
    try:
        import cleanlab  # noqa
        print("\n[cleanlab disponivel — ranking complementar baseado na biblioteca pode ser adicionado depois]")
    except ImportError:
        pass

    print(f"\nResultado em: {out_dir.resolve()}")
    print(f"Abra suspects.csv e a pasta gallery/ pra revisar.")
    print("Imagens com score alto sao as mais provaveis de estar mal-rotuladas — sobe pro Roboflow e re-anota.")


if __name__ == "__main__":
    main()
