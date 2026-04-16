"""
Download dataset de cédulas brasileiras do Roboflow.

Uso:
    1. Crie uma conta gratuita em https://roboflow.com
    2. Pegue sua API key em https://app.roboflow.com/settings/api
    3. Execute: python download_dataset.py --api-key SUA_API_KEY
"""

import argparse
import os
from roboflow import Roboflow


DATASETS = [
    {
        "workspace": "projetoiamoedas",
        "project": "moneytimes",
        "version": 1,
        "name": "MoneyTimes - Cédulas do Real",
    },
    {
        "workspace": "delphos-1tklv",
        "project": "cedulas-9fprk",
        "version": 1,
        "name": "Cedulas - Delphos",
    },
]


def download_dataset(api_key: str, dataset_idx: int = 0, output_dir: str = "datasets"):
    """Baixa um dataset do Roboflow em formato YOLOv8."""
    os.makedirs(output_dir, exist_ok=True)

    rf = Roboflow(api_key=api_key)

    ds = DATASETS[dataset_idx]
    print(f"\nBaixando: {ds['name']}")
    print(f"  Workspace: {ds['workspace']}")
    print(f"  Project:   {ds['project']}")

    project = rf.workspace(ds["workspace"]).project(ds["project"])
    version = project.version(ds["version"])
    dataset = version.download("yolov8", location=os.path.join(output_dir, ds["project"]))

    print(f"\nDataset salvo em: {dataset.location}")
    print(f"Classes: {project.classes}")
    return dataset


def list_datasets():
    """Lista os datasets disponíveis para download."""
    print("\nDatasets disponíveis:")
    for i, ds in enumerate(DATASETS):
        print(f"  [{i}] {ds['name']} ({ds['workspace']}/{ds['project']})")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download de datasets de cédulas brasileiras")
    parser.add_argument("--api-key", type=str, required=True, help="Roboflow API key")
    parser.add_argument("--dataset", type=int, default=0, help="Índice do dataset (use --list para ver)")
    parser.add_argument("--list", action="store_true", help="Lista datasets disponíveis")
    parser.add_argument("--output", type=str, default="datasets", help="Diretório de saída")
    parser.add_argument("--all", action="store_true", help="Baixa todos os datasets")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.all:
        for i in range(len(DATASETS)):
            download_dataset(args.api_key, i, args.output)
    else:
        download_dataset(args.api_key, args.dataset, args.output)
