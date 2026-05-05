"""Teste seguro de movimento da Delta.

Sequencia:
  1. Homing (G28)
  2. Desce pra Z=250 (longe da mesa)
  3. Move +15mm em X, le posicao
  4. Move -15mm em X, le posicao (volta)
  5. Move +15mm em Y, le posicao
  6. Move -15mm em Y, le posicao (volta)
  7. Sobe pra Z=290 (perto do topo, fora do caminho)
  8. Desliga motores

Movimentos pequenos (15mm) pra ficar seguro mesmo se o raio da delta for pequeno.
Velocidade baixa (1500mm/min) pra voce ver e poder cortar a fonte se algo der errado.

Uso:
    python delta_test_move.py --port COM3
"""
import argparse
import sys
import time

import serial


def send(ser, cmd, wait=2.0, timeout_extra=10):
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


def cmd(ser, c, label=None, wait=2.0):
    label = label or c
    print(f"\n>> {label}")
    resp = send(ser, c, wait=wait, timeout_extra=30)
    for line in resp.split("\n"):
        if line:
            print(f"  {line}")
    return resp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--feed", type=int, default=1500,
                        help="Velocidade em mm/min (default 1500 = devagar)")
    parser.add_argument("--step", type=int, default=15,
                        help="Tamanho do passo em mm (default 15)")
    parser.add_argument("--z-test", type=int, default=250,
                        help="Altura Z para testar movimentos (default 250)")
    args = parser.parse_args()

    print(f"Conectando em {args.port} a {args.baud} ...")
    ser = serial.Serial(args.port, args.baud, timeout=2)
    time.sleep(3)
    ser.reset_input_buffer()

    print("\n=== Iniciando teste de movimento ===")
    print(f"Velocidade: {args.feed} mm/min")
    print(f"Passo:      {args.step} mm")
    print(f"Z teste:    {args.z_test} mm")
    print("\nMantenha a mao na chave da fonte. Tem 5s pra cancelar (Ctrl+C).")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    try:
        cmd(ser, "G21", "G21 (unidades em mm)")
        cmd(ser, "G90", "G90 (coordenadas absolutas)")

        cmd(ser, "G28", "G28 (homing)", wait=30)
        cmd(ser, "M114", "M114 (posicao apos home)")

        feed = args.feed
        z = args.z_test
        s = args.step

        cmd(ser, f"G1 Z{z} F{feed}", f"Desce para Z={z}", wait=10)
        cmd(ser, "M400", "M400 (espera fim do movimento)", wait=30)
        cmd(ser, "M114", "Posicao")

        cmd(ser, f"G1 X{s} Y0 Z{z} F{feed}", f"Move X={s}", wait=8)
        cmd(ser, "M400", "espera", wait=15)
        cmd(ser, "M114", "Posicao")
        time.sleep(1)

        cmd(ser, f"G1 X-{s} Y0 Z{z} F{feed}", f"Move X=-{s}", wait=8)
        cmd(ser, "M400", "espera", wait=15)
        cmd(ser, "M114", "Posicao")
        time.sleep(1)

        cmd(ser, f"G1 X0 Y{s} Z{z} F{feed}", f"Move Y={s}", wait=8)
        cmd(ser, "M400", "espera", wait=15)
        cmd(ser, "M114", "Posicao")
        time.sleep(1)

        cmd(ser, f"G1 X0 Y-{s} Z{z} F{feed}", f"Move Y=-{s}", wait=8)
        cmd(ser, "M400", "espera", wait=15)
        cmd(ser, "M114", "Posicao")
        time.sleep(1)

        cmd(ser, f"G1 X0 Y0 Z{z} F{feed}", "Volta para o centro", wait=8)
        cmd(ser, "M400", "espera", wait=15)
        cmd(ser, "M114", "Posicao final")

        cmd(ser, f"G1 X0 Y0 Z290 F{feed}", "Sobe para Z=290 (descansa)", wait=10)
        cmd(ser, "M400", "espera", wait=15)

        cmd(ser, "M84", "M84 (desliga motores)")
        print("\n=== Teste concluido ===")

    except KeyboardInterrupt:
        print("\n!! Cancelado pelo usuario. Mandando M112 (parada) ...")
        ser.write(b"M112\n")
        ser.flush()
        time.sleep(0.5)
        sys.exit(1)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
