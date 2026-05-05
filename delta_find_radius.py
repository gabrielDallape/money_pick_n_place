"""Descobre o raio maximo seguro da Delta.

Estrategia:
1. Homing + descer pra Z seguro
2. Pra cada r em 20, 30, 40, ..., 120 mm:
   - Manda G1 X{r} Y0 Z=safe
   - Espera fim do movimento (M400)
   - Le posicao real (M114)
   - Se a posicao real != comandada (firmware clipou ou nao foi),
     significa que excedeu o limite. Para o teste.
3. Imprime relatorio: maior raio seguro testado.

Uso:
    python delta_find_radius.py --port COM3
"""
import argparse
import re
import sys
import time

import serial


M114_RE = re.compile(r"X:(-?\d+\.\d+)\s*Y:(-?\d+\.\d+)\s*Z:(-?\d+\.\d+)")


def send(ser, cmd, wait=2.0, timeout_extra=15):
    ser.reset_input_buffer()
    ser.write((cmd.strip() + "\n").encode("ascii"))
    ser.flush()
    start = time.time()
    lines = []
    while time.time() - start < wait + timeout_extra:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if not line:
            if time.time() - start > wait:
                break
            continue
        lines.append(line)
        if line.lower().startswith("ok"):
            break
    return "\n".join(lines)


def get_position(ser):
    """Manda M400 + M114 e parseia a posicao reportada."""
    send(ser, "M400", wait=2)
    resp = send(ser, "M114", wait=1)
    m = M114_RE.search(resp)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--feed", type=int, default=1500)
    parser.add_argument("--z-safe", type=int, default=250)
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--max", type=int, default=120,
                        help="Para quando atingir esse raio mesmo se ainda funcionar")
    parser.add_argument("--tolerance", type=float, default=1.5,
                        help="Tolerancia em mm pra considerar 'chegou la'")
    args = parser.parse_args()

    print(f"Conectando em {args.port} a {args.baud} ...")
    ser = serial.Serial(args.port, args.baud, timeout=2)
    time.sleep(3)
    ser.reset_input_buffer()

    print("\n=== Teste de raio maximo ===")
    print(f"Z seguro:    {args.z_safe} mm")
    print(f"Inicio:      {args.start} mm")
    print(f"Passo:       {args.step} mm")
    print(f"Limite:      {args.max} mm")
    print(f"Velocidade:  {args.feed} mm/min")
    print("\nMantenha a mao na chave da fonte. Tem 5s pra cancelar (Ctrl+C).")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    last_safe_radius = 0
    failed_at = None

    try:
        send(ser, "G21", wait=1)
        send(ser, "G90", wait=1)
        print("\n>> Homing ...")
        send(ser, "G28", wait=30, timeout_extra=30)

        print(f">> Descendo para Z={args.z_safe} ...")
        send(ser, f"G1 Z{args.z_safe} F{args.feed}", wait=10)
        send(ser, "M400", wait=10)

        r = args.start
        while r <= args.max:
            print(f"\n>> Testando raio = {r} mm em +X")
            send(ser, f"G1 X{r} Y0 Z{args.z_safe} F{args.feed}", wait=8)
            pos = get_position(ser)
            if pos is None:
                print("  !! Sem resposta de posicao — parando")
                failed_at = r
                break

            x, y, z = pos
            print(f"   comandado X={r}, Y=0  ->  real X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
            if abs(x - r) > args.tolerance or abs(y) > args.tolerance:
                print(f"   !! Posicao real diferente do comandado (>{args.tolerance}mm). LIMITE.")
                failed_at = r
                break

            last_safe_radius = r

            print("   Voltando para o centro ...")
            send(ser, f"G1 X0 Y0 Z{args.z_safe} F{args.feed}", wait=8)
            send(ser, "M400", wait=10)
            time.sleep(0.5)
            r += args.step

        # tambem volta ao centro e sobe pra posicao de descanso
        print("\n>> Subindo para Z=290 (descanso) ...")
        send(ser, f"G1 X0 Y0 Z290 F{args.feed}", wait=10)
        send(ser, "M400", wait=10)
        send(ser, "M84", wait=1)

        print("\n" + "=" * 50)
        print(f"  RAIO MAXIMO CONFIRMADO: {last_safe_radius} mm")
        if failed_at is not None:
            print(f"  Falhou em: {failed_at} mm")
        else:
            print(f"  Chegou ate o limite do teste sem falhar (--max={args.max})")
        print("=" * 50)
        print(f"\nSua area util e um circulo de raio {last_safe_radius}mm a partir do centro.")
        print(f"Diametro util: {2*last_safe_radius}mm  (~{2*last_safe_radius/10:.1f}cm)")

    except KeyboardInterrupt:
        print("\n!! Cancelado. Mandando M112.")
        ser.write(b"M112\n")
        time.sleep(0.5)
        sys.exit(1)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
