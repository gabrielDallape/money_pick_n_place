"""Active learning: ranqueia imagens NAO rotuladas por incerteza do modelo.
Use o ranking pra priorizar quais subir e rotular no Roboflow primeiro —
1000 fotos curadas valem mais que 5000 aleatorias.

Score de incerteza por imagem (quanto maior, mais util rotular):
- Baixa confianca da melhor predicao    -> modelo nao tem certeza
- Multipla predicoes em conflito (mesma area, classes diferentes)
- Nenhuma predicao OU muitas predicoes  -> caso fora da distribuicao
- Confianca media + variancia das predicoes

Uso:
    # ranqueia toda a pasta captures/ por incerteza, top 200
    python active_learn.py --src captures --top 200

    # ranqueia uma pasta especifica de uma sessao
    python active_learn.py --src captures/20260502_204124 --top 100

    # output salvo em runs/active_learning/<timestamp>/
"""
import argparse
import csv
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


def latest_run_best():
    import glob
    runs = sorted(glob.glob("runs/detect/v2*/weights/best.pt"))
    return runs[-1] if runs else None


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


def uncertainty_score(boxes_data):
    """boxes_data: lista de (cls_id, xyxy, conf). Retorna score de incerteza."""
    if not boxes_data:
        # nenhuma deteccao = caso fora da distribuicao
        # interessante rotular se o modelo realmente nao acha NADA
        return 0.6, "sem_deteccoes"

    confs = np.array([b[2] for b in boxes_data])
    score = 0.0
    reasons = []

    # 1) confianca baixa -> incerteza
    low_conf = float(np.mean(confs < 0.5))
    if low_conf > 0:
        score += low_conf * 0.6
        reasons.append(f"baixa_conf={low_conf:.2f}")

    # 2) varios pares com IoU alto e classes diferentes -> conflito
    n = len(boxes_data)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_data[i][0] != boxes_data[j][0] and iou(boxes_data[i][1], boxes_data[j][1]) > 0.4:
                conflicts += 1
    if conflicts > 0:
        score += conflicts * 0.4
        reasons.append(f"conflitos={conflicts}")

    # 3) muito ruido (12+ deteccoes em uma foto = provavelmente alucinando)
    if n > 12:
        score += 0.3
        reasons.append(f"muitas_deteccoes={n}")

    # 4) conf media baixa total
    mean_conf = float(np.mean(confs))
    score += (1.0 - mean_conf) * 0.3
    reasons.append(f"conf_media={mean_conf:.2f}")

    # 5) variancia alta -> mistura de certo e duvidoso
    if n >= 2:
        score += float(np.std(confs)) * 0.4

    return score, ",".join(reasons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Pasta com imagens nao rotuladas (recursivo)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--top", type=int, default=200,
                        help="Quantas imagens priorizar para anotacao")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Conf BAIXO de proposito (queremos ver tudo, ate o fraco)")
    parser.add_argument("--name", default=None)
    parser.add_argument("--copy", action="store_true",
                        help="Copia as top N para a pasta de saida (pronto pra subir no Roboflow)")
    args = parser.parse_args()

    model_path = args.model or latest_run_best()
    if model_path is None or not Path(model_path).exists():
        raise SystemExit("Modelo nao encontrado. Use --model <best.pt>")

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Pasta nao existe: {src}")

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(src.rglob(ext))
    images = sorted(set(images))
    if not images:
        raise SystemExit(f"Nenhuma imagem encontrada em {src}")
    print(f"Encontradas {len(images)} imagens em {src}")

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    print(f"Modelo: {model_path}")

    out_name = args.name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/active_learning") / out_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    scored = []
    for i, p in enumerate(images):
        if i % 200 == 0 and i > 0:
            print(f"  {i}/{len(images)}")
        try:
            res = model.predict(str(p), conf=args.conf, verbose=False)[0]
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
            continue
        boxes_data = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                boxes_data.append((
                    int(b.cls[0]),
                    tuple(int(v) for v in b.xyxy[0].tolist()),
                    float(b.conf[0]),
                ))
        score, reason = uncertainty_score(boxes_data)
        scored.append({
            "path": str(p),
            "score": score,
            "reason": reason,
            "n_dets": len(boxes_data),
        })

    scored.sort(key=lambda r: r["score"], reverse=True)
    top = scored[: args.top]

    csv_path = out_dir / "ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "score", "n_dets", "reason", "path"])
        for rank, r in enumerate(scored, 1):
            writer.writerow([rank, f"{r['score']:.3f}", r["n_dets"], r["reason"], r["path"]])

    if args.copy:
        priority_dir = out_dir / "priority_to_label"
        priority_dir.mkdir()
        for rank, r in enumerate(top, 1):
            src_p = Path(r["path"])
            dst = priority_dir / f"{rank:04d}_{src_p.name}"
            shutil.copy2(src_p, dst)
        print(f"\nTop {len(top)} copiadas para: {priority_dir.resolve()}")
        print("Sobe essa pasta no Roboflow — sao as imagens que mais vao ensinar o modelo.")

    print(f"\nRanking completo: {csv_path.resolve()}")
    print(f"Top {len(top)} imagens com maior incerteza:")
    for rank, r in enumerate(top[:10], 1):
        print(f"  {rank:2d}. score={r['score']:.2f}  {Path(r['path']).name}  ({r['reason']})")


if __name__ == "__main__":
    main()
