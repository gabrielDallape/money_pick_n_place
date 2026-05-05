"""Diagnostico do que acontece durante e apos o homing.

Estrategia:
1. Mostra firmware info (M115)
2. Tenta dumpar settings (M503) - alguns firmwares antigos respondem
3. Manda G28 e MEDE o tempo
4. Apos G28, faz sampling rapido de M114 (a cada 150ms por 3s)
   pra detectar qualquer motion residual
5. Manda G91/G1 Z-5/G90 e faz sampling rapido durante o move
6. Log completo no terminal pra analisar

NAO mexa enquanto roda. So observa.

Uso:
    python delta_diagnose_home.py --port COM3
"""
import argparse
import re
import time

import serial


M114_RE = re.compile(r"X:(-?\d+\.\d+)\s*Y:(-?\d+\.\d+)\s*Z:(-?\d+\.\d+).*Count\s*X:\s*(-?\d+\.\d+)\s*Y:\s*(-?\d+\.\d+)\s*Z:\s*(-?\d+\.\d+)")


def send_raw(ser, cmd, max_wait=5.0):
    """Manda comando e devolve TODAS as linhas ate ver 'ok' ou timeout."""
    ser.reset_input_buffer()
    ser.write((cmd.strip() + "\n").encode("ascii"))
    ser.flush()
    start = time.time()
    lines = []
    while time.time() - start < max_wait:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if line:
            lines.append((time.time() - start, line))
            if line.lower().startswith("ok"):
                break
    return lines


def quick_pos(ser):
    """Pega posicao rapidamente, sem print."""
    ser.reset_input_buffer()
    ser.write(b"M114\n")
    ser.flush()
    start = time.time()
    raw = ""
    while time.time() - start < 1.5:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if line:
            raw += " " + line
            if line.lower().startswith("ok"):
                break
    m = M114_RE.search(raw)
    if not m:
        return None
    return tuple(float(m.group(i)) for i in range(1, 7))


def sample_position(ser, label, duration=3.0, interval=0.15):
    """Sampleia M114 rapidamente e mostra mudancas."""
    print(f"\n--- Sampling pos durante {duration}s ({label}) ---")
    print("    t(s)   X       Y       Z       cX       cY       cZ")
    last = None
    start = time.time()
    while time.time() - start < duration:
        p = quick_pos(ser)
        t = time.time() - start
        if p is None:
            print(f"   {t:5.2f}  (sem resposta)")
        else:
            x, y, z, cx, cy, cz = p
            mark = ""
            if last is not None:
                if abs(z - last[2]) > 0.01:
                    mark = "   <-- Z mudou!"
                if abs(cx - last[3]) > 0.01 or abs(cy - last[4]) > 0.01 or abs(cz - last[5]) > 0.01:
                    if not mark:
                        mark = "   <-- count mudou!"
            print(f"   {t:5.2f}  {x:6.2f}  {y:6.2f}  {z:6.2f}   {cx:7.2f}  {cy:7.2f}  {cz:7.2f}{mark}")
            last = p
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    print(f"Conectando em {args.port} ...")
    ser = serial.Serial(args.port, args.baud, timeout=2)
    time.sleep(3)
    ser.reset_input_buffer()

    print("\n=== M115 (firmware) ===")
    for t, line in send_raw(ser, "M115"):
        print(f"   {line}")

    print("\n=== M503 (settings — pode nao responder em firmware antigo) ===")
    for t, line in send_raw(ser, "M503", max_wait=3):
        print(f"   {line}")

    print("\n=== G21 G90 (init) ===")
    send_raw(ser, "G21", max_wait=1)
    send_raw(ser, "G90", max_wait=1)

    print("\n=== POSICAO ANTES DO HOMING ===")
    p = quick_pos(ser)
    print(f"   {p}")

    print("\n=== G28 (homing) — vai mexer ===")
    print("   tempo  | resposta")
    t0 = time.time()
    for t, line in send_raw(ser, "G28", max_wait=60):
        print(f"   {t:5.2f}s | {line}")
    g28_time = time.time() - t0
    print(f"\n   G28 levou: {g28_time:.2f}s")

    # Sampling logo apos o ok do G28
    sample_position(ser, "POS apos G28 OK", duration=3.0, interval=0.15)

    print("\n=== M400 (sync) ===")
    for t, line in send_raw(ser, "M400"):
        print(f"   {line}")
    sample_position(ser, "POS apos M400", duration=2.0, interval=0.15)

    print("\n=== G91 + G1 Z-5 F1000 + G90 (descida 5mm relativa) ===")
    send_raw(ser, "G91")
    print("   G91 ok")

    # mando o G1 e ja comeco a samplear durante o movimento
    print("\n   --- comeco do G1 Z-5 ---")
    ser.reset_input_buffer()
    ser.write(b"G1 Z-5 F1000\n")
    ser.flush()

    move_start = time.time()
    print("    t(s)   X       Y       Z       cX       cY       cZ      [resposta]")
    last_p = None
    last_response_check = move_start
    while time.time() - move_start < 5:
        # tenta ler linha do firmware sem bloquear muito
        ser.timeout = 0.05
        line = ser.readline().decode("ascii", errors="replace").strip()
        ser.timeout = 2
        if line:
            print(f"   {time.time()-move_start:5.2f}                                                              {line}")
            if line.lower().startswith("ok"):
                # depois do ok, faz mais alguns samples e para
                pass

        # sample rapido
        if time.time() - last_response_check > 0.1:
            p = quick_pos(ser)
            t = time.time() - move_start
            if p:
                x, y, z, cx, cy, cz = p
                mark = ""
                if last_p is not None:
                    dz = z - last_p[2]
                    if dz > 0.05:
                        mark = "   <-- Z SUBIU!"
                    elif dz < -0.05:
                        mark = "   <-- Z desceu"
                print(f"   {t:5.2f}  {x:6.2f}  {y:6.2f}  {z:6.2f}   {cx:7.2f}  {cy:7.2f}  {cz:7.2f}{mark}")
                last_p = p
            last_response_check = time.time()

    send_raw(ser, "G90")
    send_raw(ser, "M84")
    ser.close()
    print("\n=== Diagnostico completo. ===")


if __name__ == "__main__":
    main()
