"""Captura imagens da webcam para popular dataset (upload manual no Roboflow depois).

Uso:
    python capture_dataset.py
    python capture_dataset.py --auto 1.5            # auto-captura a cada 1.5s
    python capture_dataset.py --out captures --tag mesa_madeira
    python capture_dataset.py --cam 1 --width 1920 --height 1080

Atalhos na janela:
    SPACE  -> captura 1 frame
    A      -> liga/desliga auto-captura
    +/-    -> ajusta intervalo da auto-captura
    B      -> liga/desliga filtro de blur (descarta fotos tremidas)
    R      -> nova sessao (zera contador)
    Q      -> sair

Dicas para um bom dataset:
    - varie iluminacao (sol, lampada, sombra)
    - varie fundo (mesa madeira, pano escuro, papel branco, chao)
    - varie angulo (cima, lateral, oblique)
    - inclua notas dobradas, amassadas, sobrepostas
    - inclua varias notas no mesmo frame
    - fotos sem nenhuma nota tambem ajudam (negativos)
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np


BLUR_MIN = 80.0  # Laplacian variance abaixo disso = imagem tremida


def open_camera(cam_idx, width, height):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0] if hasattr(cv2, "CAP_DSHOW") else [0]
    for backend in backends:
        cap = cv2.VideoCapture(cam_idx, backend) if backend else cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        cap.release()
    return None


def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def make_session_dir(out_root: Path, tag: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    session = out_root / name
    session.mkdir(parents=True, exist_ok=True)
    return session


def draw_hud(img, lines, x=10, y0=30):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (text, color) in enumerate(lines):
        y = y0 + i * 32
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
        cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 8, y + 6), (20, 20, 20), -1)
        cv2.putText(img, text, (x, y), font, 0.6, color, 2, cv2.LINE_AA)


def flash(img, alpha=0.6):
    overlay = np.full_like(img, 255)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--out", type=str, default="captures")
    parser.add_argument("--tag", type=str, default="", help="Sufixo no nome da sessao (ex: mesa_madeira)")
    parser.add_argument("--auto", type=float, default=0.0,
                        help="Se > 0, captura automatica a cada N segundos")
    parser.add_argument("--no-blur-filter", action="store_true",
                        help="Desabilita filtro de blur na auto-captura")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    session = make_session_dir(out_root, args.tag)
    print(f"Sessao: {session.resolve()}")

    cap = open_camera(args.cam, args.width, args.height)
    if cap is None or not cap.isOpened():
        print(f"Falha ao abrir camera {args.cam}")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolucao: {actual_w}x{actual_h}")

    win = "Capture Dataset - Money Pick & Place"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    auto_interval = float(args.auto)
    auto_on = auto_interval > 0
    blur_filter = not args.no_blur_filter
    last_auto = time.time()
    flash_until = 0.0
    skipped_blur = 0
    counter = 0

    save_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(50, min(100, args.jpeg_quality))]

    print("SPACE: captura | A: auto | +/-: intervalo | B: blur filter | R: nova sessao | Q: sair")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler frame")
            break

        now = time.time()
        do_capture = False

        # auto-captura
        if auto_on and (now - last_auto) >= auto_interval:
            last_auto = now
            if blur_filter:
                if blur_score(frame) >= BLUR_MIN:
                    do_capture = True
                else:
                    skipped_blur += 1
            else:
                do_capture = True

        # tela: copia para nao desenhar HUD em cima do salvo
        display = frame.copy()
        if now < flash_until:
            display = flash(display, alpha=0.5)

        bs = blur_score(frame)
        bs_color = (0, 255, 120) if bs >= BLUR_MIN else (0, 100, 255)

        hud = [
            (f"Capturadas: {counter}", (255, 255, 255)),
            (f"Sessao: {session.name}", (200, 200, 200)),
            (f"Auto: {'ON' if auto_on else 'OFF'}  (a cada {auto_interval:.1f}s)" if auto_on
             else "Auto: OFF", (0, 220, 255) if auto_on else (180, 180, 180)),
            (f"Blur filter: {'ON' if blur_filter else 'OFF'}  (score={bs:.0f})", bs_color),
            (f"Pulados por blur: {skipped_blur}", (180, 180, 180)),
            (f"Resolucao: {actual_w}x{actual_h}", (180, 180, 180)),
        ]
        draw_hud(display, hud)

        # legenda inferior
        h = display.shape[0]
        cv2.rectangle(display, (0, h - 36), (display.shape[1], h), (20, 20, 20), -1)
        cv2.putText(display, "[SPACE] capturar  [A] auto  [+/-] intervalo  [B] blur  [R] nova sessao  [Q] sair",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            do_capture = True
        elif key == ord("a"):
            auto_on = not auto_on
            if auto_on and auto_interval <= 0:
                auto_interval = 1.5
            last_auto = now
        elif key in (ord("+"), ord("=")):
            auto_interval = min(auto_interval + 0.5, 30.0)
        elif key == ord("-"):
            auto_interval = max(auto_interval - 0.5, 0.5)
        elif key == ord("b"):
            blur_filter = not blur_filter
        elif key == ord("r"):
            print(f"Sessao anterior: {counter} fotos em {session}")
            session = make_session_dir(out_root, args.tag)
            counter = 0
            skipped_blur = 0
            print(f"Nova sessao: {session.resolve()}")

        if do_capture:
            counter += 1
            ts = int(time.time() * 1000)
            fname = session / f"img_{counter:05d}_{ts}.jpg"
            cv2.imwrite(str(fname), frame, save_params)
            flash_until = now + 0.12
            if counter % 10 == 0 or not auto_on:
                print(f"[{counter:5d}] salvo {fname.name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFim. {counter} foto(s) em: {session.resolve()}")
    print("Proximo passo: subir essa pasta no Roboflow (workspace 'detectorobjetos', projeto 'pick-n-place-money')")


if __name__ == "__main__":
    main()
