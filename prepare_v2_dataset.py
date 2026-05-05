"""Prepara o dataset v2 vindo do Roboflow:
- Le datasets/v2_raw/ (que tem so split 'train/')
- Agrupa imagens por foto-fonte (rotacoes da mesma foto vao juntas) pra evitar leakage
- Divide em train/valid/test (default 80/15/5)
- Move arquivos pra datasets/v2/{train,valid,test}/{images,labels}/
- Escreve datasets/v2/data.yaml

Uso:
    python prepare_v2_dataset.py
    python prepare_v2_dataset.py --src datasets/v2_raw --dst datasets/v2 --val 0.15 --test 0.05
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


ROT_PREFIX_RE = re.compile(r"^(?:0|90|180|270)_")
RF_HASH_RE = re.compile(r"\.rf\.[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*$")


def group_key(stem: str) -> str:
    """Reduz o nome do arquivo a uma chave de foto-fonte.
    Ex.: '90_5RealBra_47_jpeg_jpg.rf.LVcZRT36...' -> '5RealBra_47_jpeg_jpg'
         '0_5RealBra_47_jpeg.rf.KB9...'           -> '5RealBra_47_jpeg'
    """
    s = RF_HASH_RE.sub("", stem)
    s = ROT_PREFIX_RE.sub("", s)
    return s


def collect_pairs(src: Path):
    img_dir = src / "train" / "images"
    lbl_dir = src / "train" / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        raise SystemExit(f"Estrutura inesperada em {src}. Esperado train/images e train/labels.")

    pairs = []
    missing_lbl = 0
    for img in img_dir.iterdir():
        if not img.is_file() or img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing_lbl += 1
            continue
        pairs.append((img, lbl))
    if missing_lbl:
        print(f"[aviso] {missing_lbl} imagem(ns) sem label correspondente foram ignoradas.")
    return pairs


def split_by_group(pairs, val_frac, test_frac, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for img, lbl in pairs:
        groups[group_key(img.stem)].append((img, lbl))

    keys = list(groups.keys())
    rng.shuffle(keys)
    n = len(keys)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test:n_test + n_val])
    # train_keys implicito = restante

    splits = {"train": [], "valid": [], "test": []}
    for k, items in groups.items():
        if k in test_keys:
            splits["test"].extend(items)
        elif k in val_keys:
            splits["valid"].extend(items)
        else:
            splits["train"].extend(items)
    return splits, len(groups)


def move_split(items, dst_split: Path):
    img_out = dst_split / "images"
    lbl_out = dst_split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img, lbl in items:
        shutil.move(str(img), img_out / img.name)
        shutil.move(str(lbl), lbl_out / lbl.name)


def write_yaml(dst: Path, names):
    dst.mkdir(parents=True, exist_ok=True)
    yml = {
        "path": str(dst.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": list(names),
    }
    with (dst / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(yml, f, sort_keys=False, allow_unicode=True)


def read_classes(src: Path):
    with (src / "data.yaml").open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return d["names"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="datasets/v2_raw")
    parser.add_argument("--dst", default="datasets/v2")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} ja existe e nao esta vazio. Apague antes ou use outro --dst.")

    names = read_classes(src)
    print(f"Classes ({len(names)}): {names}")

    pairs = collect_pairs(src)
    print(f"Pares imagem+label encontrados: {len(pairs)}")

    splits, n_groups = split_by_group(pairs, args.val, args.test, args.seed)
    print(f"Grupos (fotos-fonte distintas): {n_groups}")
    for s, items in splits.items():
        print(f"  {s}: {len(items)} imagens")

    for s, items in splits.items():
        move_split(items, dst / s)

    write_yaml(dst, names)
    print(f"\nOK. Dataset preparado em: {dst.resolve()}")
    print(f"data.yaml escrito. Pode treinar agora com: python train_v2.py")


if __name__ == "__main__":
    main()
