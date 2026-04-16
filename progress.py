"""Mostra barra de progresso do treino YOLO em andamento."""
import re
import sys
import time
from pathlib import Path

RUN_DIR = Path("runs/detect/runs/cedulas_gpu")
LOG_FILE = Path("training_resume.log")
TOTAL_EPOCHS = 150
BAR_WIDTH = 40


def parse_state():
    csv = RUN_DIR / "results.csv"
    completed = 0
    times = []
    if csv.exists():
        lines = csv.read_text().strip().splitlines()
        if len(lines) > 1:
            data = [l.split(",") for l in lines[1:]]
            completed = len(data)
            times = [float(r[1]) for r in data if r[1]]

    cur_epoch = completed + 1
    cur_pct = 0.0
    if LOG_FILE.exists():
        text = LOG_FILE.read_text(errors="ignore")
        matches = re.findall(r"(\d+)/" + str(TOTAL_EPOCHS) + r"\s+[\d.]+G.*?(\d+)/(\d+)", text)
        if matches:
            ep, it, total = matches[-1]
            cur_epoch = int(ep)
            cur_pct = int(it) / int(total)

    global_done = completed + cur_pct
    global_pct = global_done / TOTAL_EPOCHS

    avg = None
    if len(times) >= 2:
        deltas = [times[i] - times[i - 1] for i in range(1, len(times))]
        avg = sum(deltas[-5:]) / len(deltas[-5:])
    eta_min = None
    if avg:
        remaining = (TOTAL_EPOCHS - global_done) * avg
        eta_min = remaining / 60

    return {
        "completed": completed,
        "cur_epoch": cur_epoch,
        "cur_pct": cur_pct,
        "global_pct": global_pct,
        "eta_min": eta_min,
        "avg_epoch_s": avg,
    }


def bar(pct, width=BAR_WIDTH):
    filled = int(round(pct * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render(s):
    print("\n=== Treino YOLOv8 - cedulas ===")
    print(f"{bar(s['global_pct'])} {s['global_pct']*100:5.1f}%")
    print(f"   Epoca {s['cur_epoch']}/{TOTAL_EPOCHS}  "
          f"(completas: {s['completed']}, atual: {s['cur_pct']*100:.0f}%)")
    if s["avg_epoch_s"]:
        print(f"   ~{s['avg_epoch_s']:.0f}s/epoca  |  ETA: ~{s['eta_min']:.0f} min "
              f"(~{s['eta_min']/60:.1f} h)")
    print()


if __name__ == "__main__":
    watch = "--watch" in sys.argv
    if watch:
        try:
            while True:
                s = parse_state()
                render(s)
                if s["global_pct"] >= 1.0:
                    print("Treino concluido.")
                    break
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nMonitoramento interrompido.")
    else:
        render(parse_state())
