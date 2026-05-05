"""Cria um 'golden holdout set' a partir do split test/ atual e CONGELA.

Boa pratica de empresa: alem do val (early stopping) e test (checagem ocasional),
voce mantem um conjunto que NUNCA e olhado durante iteracao. So abre na hora de
fechar uma versao do modelo. Sem isso, voce acaba (mesmo sem querer) otimizando
hyperparams pro test set e a metrica final mente.

O que esse script faz:
- Move N imagens (default 50) do test/ pra holdout/
- Calcula SHA256 de cada arquivo
- Escreve HOLDOUT_LOCK.json e HOLDOUT_LOCK.md com a data, seed, hashes
- Atualiza data.yaml com o novo path 'holdout'

Uso (uma vez so):
    python freeze_holdout.py
    python freeze_holdout.py --n 80 --seed 13

Apos rodar:
    - NAO rode evaluate_model.py --split holdout durante iteracao normal
    - So abra esse split em milestones (release v2, release v3, etc)
    - Comite o HOLDOUT_LOCK.md no git pra audit trail
"""
import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/v2/data.yaml")
    parser.add_argument("--from-split", default="test",
                        help="Split de origem (default: test)")
    parser.add_argument("--n", type=int, default=50,
                        help="Quantas imagens congelar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="holdout")
    parser.add_argument("--reason", default="Golden set v2 - congelado para release v2.0",
                        help="Motivo registrado no lock file")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(f"data.yaml nao existe: {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        ymeta = yaml.safe_load(f)

    base = Path(ymeta.get("path", data_path.parent)).resolve()
    src_img_dir = (base / args.from_split / "images")
    src_lbl_dir = (base / args.from_split / "labels")
    if not src_img_dir.exists():
        raise SystemExit(f"Split origem nao encontrado: {src_img_dir}")

    holdout_root = base / args.name
    if holdout_root.exists():
        raise SystemExit(f"{holdout_root} ja existe. Holdout ja foi criado — NAO refaca.\n"
                         f"Se realmente precisa recriar, apague manualmente e justifique no commit.")
    (holdout_root / "images").mkdir(parents=True)
    (holdout_root / "labels").mkdir(parents=True)

    images = sorted([p for p in src_img_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if len(images) < args.n:
        raise SystemExit(f"Split origem tem {len(images)} imagens, menos que --n={args.n}")

    rng = random.Random(args.seed)
    chosen = rng.sample(images, args.n)
    print(f"Movendo {len(chosen)} imagens de {args.from_split}/ -> {args.name}/ (seed={args.seed})")

    locked = []
    for img in chosen:
        lbl = src_lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            print(f"[skip] sem label: {img.name}")
            continue
        new_img = holdout_root / "images" / img.name
        new_lbl = holdout_root / "labels" / lbl.name
        shutil.move(str(img), new_img)
        shutil.move(str(lbl), new_lbl)
        locked.append({
            "image": img.name,
            "label": lbl.name,
            "image_sha256": sha256_file(new_img),
            "label_sha256": sha256_file(new_lbl),
        })

    # atualiza data.yaml com o split holdout
    ymeta[args.name] = f"{args.name}/images"
    with data_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(ymeta, f, sort_keys=False, allow_unicode=True)

    lock_data = {
        "name": args.name,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "from_split": args.from_split,
        "seed": args.seed,
        "count": len(locked),
        "reason": args.reason,
        "data_yaml": str(data_path),
        "files": locked,
    }
    lock_json = base / "HOLDOUT_LOCK.json"
    lock_json.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")

    md_lines = [
        f"# Holdout set congelado — {args.name}",
        "",
        f"- **Congelado em:** {lock_data['frozen_at']}",
        f"- **Origem:** split `{args.from_split}`",
        f"- **Seed:** {args.seed}",
        f"- **Total de imagens:** {len(locked)}",
        f"- **Motivo:** {args.reason}",
        "",
        "## Regras",
        "",
        "1. **NAO** olhe esse split durante iteracao. So no fechamento de uma versao do modelo.",
        "2. Se rodar `evaluate_model.py` nele mais de uma vez por release, a metrica deixa de ser",
        "   independente — voce comeca a otimizar pra ele indiretamente.",
        "3. Para verificar integridade:",
        "   ```bash",
        f"   python -c \"import json,hashlib,pathlib;"
        f"d=json.load(open('HOLDOUT_LOCK.json'));"
        f"print(all(hashlib.sha256(open(pathlib.Path('{args.name}/images')/f['image'],'rb').read()).hexdigest()==f['image_sha256'] for f in d['files']))\"",
        "   ```",
        "4. Comite esse arquivo no git pra audit trail.",
        "",
        "## Arquivos congelados",
        "",
        "| # | Imagem | SHA256 (8 chars) |",
        "|---|--------|------------------|",
    ]
    for i, item in enumerate(locked, 1):
        md_lines.append(f"| {i} | {item['image']} | `{item['image_sha256'][:8]}` |")

    (base / "HOLDOUT_LOCK.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nHoldout congelado:")
    print(f"  Pasta:    {holdout_root}")
    print(f"  Lock:     {lock_json.name}")
    print(f"  Doc:      HOLDOUT_LOCK.md")
    print(f"  data.yaml atualizado: split '{args.name}' adicionado")
    remaining = len(list((base / args.from_split / 'images').iterdir()))
    print(f"\n  Restante em {args.from_split}/: {remaining} imagens")
    print(f"\n  REGRA: NAO use --split {args.name} ate fechar o release v2.")


if __name__ == "__main__":
    main()
