"""Inspecao visual do dataset com FiftyOne.

Carrega o dataset YOLO em FiftyOne e abre uma UI no browser onde voce pode:
- ver bbox por classe e filtrar
- buscar por confianca, tamanho, area
- comparar predicoes do modelo com ground truth (achar erros)
- marcar amostras com tag para revisao no Roboflow

Uso:
    python inspect_dataset.py                              # so ground truth
    python inspect_dataset.py --model runs/detect/v2_*/weights/best.pt   # com predicoes
    python inspect_dataset.py --split valid                # so o split valid
    python inspect_dataset.py --persistent                 # mantem dataset entre execucoes

Apos abrir, vai imprimir um link tipo http://localhost:5151. Abre no browser.
"""
import argparse
from pathlib import Path

import yaml


def load_yolo_split(name, data_yaml, dataset_dir):
    import fiftyone as fo

    split_dir = dataset_dir / name
    img_dir = split_dir / "images"
    if not img_dir.exists():
        return None

    print(f"Carregando split '{name}' de {img_dir}...")
    ds = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset,
        yaml_path=str(data_yaml),
        split=name,
        label_field="ground_truth",
    )
    ds.tags.append(name)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/v2/data.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    parser.add_argument("--model", default=None,
                        help="Opcional: pesos .pt para sobrepor predicoes ao ground truth")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--name", default="money_v2",
                        help="Nome do dataset no FiftyOne")
    parser.add_argument("--persistent", action="store_true",
                        help="Mantem o dataset salvo entre execucoes")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limita quantas imagens carregar (debug)")
    args = parser.parse_args()

    try:
        import fiftyone as fo
    except ImportError:
        raise SystemExit("fiftyone nao instalado. pip install fiftyone")

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml nao encontrado: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        ymeta = yaml.safe_load(f)
    dataset_dir = data_yaml.parent
    print(f"Dataset: {dataset_dir}")
    print(f"Classes: {ymeta['names']}")

    # remove dataset anterior com mesmo nome
    if args.name in fo.list_datasets():
        fo.delete_dataset(args.name)

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]

    main_ds = fo.Dataset(args.name, persistent=args.persistent)

    for sp in splits:
        sub = load_yolo_split(sp, data_yaml, dataset_dir)
        if sub is None:
            continue
        if args.max_samples:
            sub = sub.limit(args.max_samples).clone()
        main_ds.merge_samples(sub)
        try:
            fo.delete_dataset(sub.name)
        except Exception:
            pass

    print(f"Total de amostras carregadas: {len(main_ds)}")

    # opcional: sobrepor predicoes do modelo
    if args.model:
        from ultralytics import YOLO
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"[aviso] Modelo nao encontrado: {model_path} — pulando predicoes.")
        else:
            print(f"Rodando inferencia com {model_path} ...")
            yolo = YOLO(str(model_path))
            class_names = ymeta["names"]
            for sample in main_ds.iter_samples(progress=True):
                res = yolo.predict(sample.filepath, conf=args.conf, verbose=False)[0]
                detections = []
                if res.boxes is not None and len(res.boxes) > 0:
                    h, w = res.orig_shape
                    for b in res.boxes:
                        cls_id = int(b.cls[0])
                        confidence = float(b.conf[0])
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        # FiftyOne usa bbox normalizada [x, y, w, h] no espaco [0,1]
                        bbox = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
                        detections.append(fo.Detection(
                            label=class_names[cls_id],
                            bounding_box=bbox,
                            confidence=confidence,
                        ))
                sample["predictions"] = fo.Detections(detections=detections)
                sample.save()

            # avalia predicoes vs ground_truth
            print("Avaliando predicoes...")
            try:
                main_ds.evaluate_detections(
                    "predictions",
                    gt_field="ground_truth",
                    eval_key="eval",
                    compute_mAP=True,
                )
                print("Eval salvo no campo 'eval'. Filtre por eval_fp/eval_fn na UI.")
            except Exception as e:
                print(f"[aviso] evaluate_detections falhou: {e}")

    print("\nAbrindo FiftyOne app no browser...")
    print("Dicas:")
    print("  - Filtra por 'tags' (train/valid/test)")
    print("  - Filtra por 'predictions.detections.label' pra ver uma classe especifica")
    if args.model:
        print("  - Use 'eval_fp' / 'eval_fn' pra ver onde o modelo erra")
        print("  - Tagueia amostras suspeitas com 'review' pra revisar depois")
    print("  - Ctrl+C aqui pra fechar")

    session = fo.launch_app(main_ds)
    session.wait()


if __name__ == "__main__":
    main()
