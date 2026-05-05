"""Primeiro contato com a impressora Delta via USB serial.

O que esse script faz:
1. Conecta na porta COM
2. Pergunta o firmware (M115) — confirma que e Marlin
3. Pergunta a posicao atual (M114)
4. (Opcional) Faz homing (G28) — SO se a fonte estiver ligada e o caminho livre

Uso:
    python delta_hello.py                    # so info, nao move nada
    python delta_hello.py --home             # info + homing (CUIDADO)
    python delta_hello.py --port COM4 --baud 250000

Bauds tipicos: 115200 (mais comum em Marlin), 250000 (algumas configs custom).
Se nao funcionar com 115200, tenta 250000.
"""
import argparse
import sys
import time

import serial


def send(ser: serial.Serial, cmd: str, wait: float = 1.0) -> str:
    """Manda um comando e le a resposta ate ver 'ok' ou timeout."""
    ser.reset_input_buffer()
    ser.write((cmd.strip() + "\n").encode("ascii"))
    ser.flush()

    start = time.time()
    lines = []
    while time.time() - start < wait + 5:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if not line:
            if time.time() - start > wait:
                break
            continue
        lines.append(line)
        if line.lower().startswith("ok"):
            break
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--home", action="store_true",
                        help="Manda G28 (CUIDADO: ela vai se mexer)")
    parser.add_argument("--baud-fallback", type=int, default=250000,
                        help="Se --baud nao funcionar, tenta esse")
    args = parser.parse_args()

    print(f"Tentando abrir {args.port} a {args.baud} baud ...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
    except serial.SerialException as e:
        print(f"Falha ao abrir {args.port}: {e}")
        print("Verifique:")
        print(" - Outro programa (Cura, Repetier, OctoPrint) esta usando a porta?")
        print(" - O cabo USB esta conectado?")
        sys.exit(1)

    # Marlin reseta a placa quando voce abre a serial. Espera a placa bootar.
    print("Aguardando boot da placa (3s) ...")
    time.sleep(3)
    ser.reset_input_buffer()

    # Tenta uma comunicacao basica
    print("\n--- M115 (firmware info) ---")
    resp = send(ser, "M115", wait=2)
    print(resp if resp else "(sem resposta — talvez baud errado)")

    if not resp:
        ser.close()
        print(f"\nNada veio a {args.baud}. Tentando {args.baud_fallback} ...")
        ser = serial.Serial(args.port, args.baud_fallback, timeout=2)
        time.sleep(3)
        ser.reset_input_buffer()
        resp = send(ser, "M115", wait=2)
        print(resp if resp else "(ainda nada)")
        if not resp:
            print("\nFalhou nos dois bauds. Possiveis causas:")
            print(" - Placa esta congelada → desligue a fonte por 5s e tente de novo")
            print(" - Outro baudrate (1200, 9600, 250000)")
            print(" - Firmware quebrado")
            ser.close()
            sys.exit(2)

    print("\n--- M114 (posicao atual) ---")
    print(send(ser, "M114", wait=1))

    if args.home:
        print("\n--- G28 (HOMING — ela vai se mexer agora) ---")
        print("Tem 5s pra cancelar (Ctrl+C) ou desligar a fonte.")
        for i in range(5, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        print("Mandando G28 ...")
        print(send(ser, "G28", wait=30))    # homing pode demorar
        print("\n--- M114 (posicao depois do home) ---")
        print(send(ser, "M114", wait=1))

    print("\n--- M84 (desliga os motores — pode mover na mao) ---")
    print(send(ser, "M84", wait=1))

    ser.close()
    print("\nConexao fechada. Tudo certo!")


if __name__ == "__main__":
    main()
