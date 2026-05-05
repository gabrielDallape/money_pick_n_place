"""Treino v2 com transfer learning, tracking W&B, seeds deterministicos e log de config.

Uso:
    python train_v2.py
    python train_v2.py --epochs 150 --imgsz 800 --batch 16
    python train_v2.py --from-scratch --model yolov8s.pt
    python train_v2.py --no-wandb           # desliga tracking remoto

W&B:
    Primeira vez: rode `wandb login` no terminal.
    Depois disso, todo treino fica visivel em wandb.ai/<seu-user>/money-pick-n-place.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO


V1_BEST = "runs/detect/runs/cedulas2/weights/best.pt"
DEFAULT_DATA = "datasets/v2/data.yaml"
PROJECT_DIR = "runs/detect"
WANDB_PROJECT = "money-pick-n-place"


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cudnn deterministico (~5-10% mais lento, mas reproduzivel)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_name(prefix="v2"):
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def setup_wandb(run_name, config, enabled):
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("[aviso] wandb nao instalado. pip install wandb. Continuando sem tracking.")
        return None
    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config=config,
            reinit=True,
        )
        return run
    except Exception as e:
        print(f"[aviso] wandb.init falhou ({e}). Rode 'wandb login' ou use --no-wandb.")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=V1_BEST,
                        help="Pesos iniciais (default: best.pt do v1 = transfer learning)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Ignora --model e usa yolov8n.pt (sem transfer do v1)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    parser.add_argument("--name", default=None, help="Nome do run (default: v2_<timestamp>)")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr0", type=float, default=0.005)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"data.yaml nao existe: {args.data}\nRode primeiro: python prepare_v2_dataset.py")

    set_seeds(args.seed)

    model_path = "yolov8n.pt" if args.from_scratch else args.model
    if not args.from_scratch and not Path(model_path).exists():
        print(f"[aviso] {model_path} nao encontrado. Caindo para yolov8n.pt.")
        model_path = "yolov8n.pt"

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("[aviso] Treinando em CPU vai demorar MUITO. Considere --epochs 30 pra teste.")

    run_name = args.name or make_run_name()

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=PROJECT_DIR,
        name=run_name,
        exist_ok=False,
        patience=args.patience,
        save=True,
        plots=True,
        seed=args.seed,
        deterministic=True,
        optimizer="auto",
        lr0=args.lr0,
        cos_lr=True,
        # augmentacao agressiva pra generalizar pra fundos novos (problema do v1)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=15.0, translate=0.1, scale=0.5, shear=2.0,
        fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.0,
        erasing=0.4,
    )

    config = {
        "model_init": model_path,
        "from_scratch": args.from_scratch,
        "v1_baseline": V1_BEST,
        **train_kwargs,
    }

    print("=" * 60)
    print(f"Run name:       {run_name}")
    print(f"Modelo inicial: {model_path}")
    print(f"Dataset:        {args.data}")
    print(f"Device:         {device}")
    print(f"Seed:           {args.seed}")
    print("=" * 60)

    wandb_run = setup_wandb(run_name, config, enabled=not args.no_wandb)

    yolo = YOLO(model_path)

    # injeta callback para mandar metricas de cada epoca pro W&B
    if wandb_run is not None:
        import wandb

        def on_epoch_end(trainer):
            try:
                metrics = {k: float(v) for k, v in trainer.metrics.items() if isinstance(v, (int, float))}
                metrics["epoch"] = trainer.epoch
                wandb.log(metrics, step=trainer.epoch)
            except Exception as e:
                print(f"[wandb] log falhou: {e}")

        yolo.add_callback("on_fit_epoch_end", on_epoch_end)

    results = yolo.train(**train_kwargs)

    # ultralytics decide o save_dir real (pode estar nesteado em runs/detect/runs/detect/...)
    try:
        run_dir = Path(yolo.trainer.save_dir)
    except Exception:
        run_dir = Path(PROJECT_DIR) / run_name
    best = run_dir / "weights" / "best.pt"
    config_path = run_dir / "train_config.json"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)
    except Exception as e:
        print(f"[aviso] nao consegui salvar train_config.json: {e}")

    print("\n" + "=" * 60)
    print(f"Treino concluido. Melhor modelo: {best.resolve()}")
    print(f"Config salva em: {config_path}")
    if wandb_run is not None:
        print(f"W&B run: {wandb_run.url}")
    print("\nProximos passos:")
    print(f"  python evaluate_model.py --model {best.as_posix()} --data {args.data}")
    print(f"  python inspect_dataset.py --data {args.data} --model {best.as_posix()}")
    print(f"  python realtime_test.py --model {best.as_posix()}")
    print("=" * 60)

    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
