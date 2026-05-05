"""Controle manual da Delta com botoes clicaveis (OpenCV).

Layout: HUD em cima + grid de botoes embaixo.
Clica num botao pra acao acontecer. Tambem aceita as mesmas teclas se preferir.

Uso:
    python delta_jog.py --port COM3
"""
import argparse
import re
import sys
import time

import cv2
import numpy as np
import serial


M114_RE = re.compile(r"X:(-?\d+\.\d+)\s*Y:(-?\d+\.\d+)\s*Z:(-?\d+\.\d+)")

WIN_W, WIN_H = 760, 620


def send(ser, cmd, wait=1.0, timeout_extra=10):
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
    send(ser, "M400", wait=2)
    resp = send(ser, "M114", wait=1)
    m = M114_RE.search(resp)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


class Button:
    def __init__(self, x, y, w, h, label, action, color=(60, 60, 60), text_color=(240, 240, 240)):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label
        self.action = action
        self.color = color
        self.text_color = text_color
        self.hover = False
        self.pressed_until = 0.0

    def hit(self, mx, my):
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def draw(self, img):
        col = self.color
        if time.time() < self.pressed_until:
            col = tuple(min(c + 80, 255) for c in col)
        elif self.hover:
            col = tuple(min(c + 30, 255) for c in col)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), col, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (200, 200, 200), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thick = 2
        (tw, th), _ = cv2.getTextSize(self.label, font, scale, thick)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(img, self.label, (tx, ty), font, scale, self.text_color, thick, cv2.LINE_AA)


class JogApp:
    def __init__(self, ser, args):
        self.ser = ser
        self.args = args
        self.pos = None
        self.step = 5
        self.feed = args.feed
        self.max_radius = args.max_radius
        self.z_min = args.z_min
        self.z_max = args.z_max
        # Se invert_z=True: aumentar Z fisicamente = subir; o firmware deste
        # robo invertia (estilo Sprinter). DESCER fisica = +Z se invertido.
        self.invert_z = args.invert_z
        self.msg = "Pronto"
        self.busy = False
        self.quit = False
        self.buttons = []
        self.mouse = (-1, -1)
        self.click = None
        self._build_buttons()

    def _build_buttons(self):
        BIG = (80, 200, 80)
        DIR = (60, 80, 140)
        UTIL = (100, 60, 100)
        DANGER = (40, 40, 180)

        # cruz direcional X/Y (ocupa 280x240 a partir de x=30, y=200)
        cx, cy = 170, 320
        bw, bh = 80, 70
        gap = 10

        self.buttons.append(Button(cx - bw // 2, cy - bh - bh // 2 - gap, bw, bh, "Y+", ("rel", "y", +1), DIR))
        self.buttons.append(Button(cx - bw // 2, cy + bh // 2 + gap, bw, bh, "Y-", ("rel", "y", -1), DIR))
        self.buttons.append(Button(cx - bw - bw // 2 - gap, cy - bh // 2, bw, bh, "X-", ("rel", "x", -1), DIR))
        self.buttons.append(Button(cx + bw // 2 + gap, cy - bh // 2, bw, bh, "X+", ("rel", "x", +1), DIR))

        # SUBIR / DESCER ao lado direito do cruz (rotulos fisicos)
        zx = cx + bw + bw // 2 + gap * 2 + 60
        self.buttons.append(Button(zx, cy - bh - gap, bw, bh, "SUBIR", ("rel", "z_up",), DIR))
        self.buttons.append(Button(zx, cy + gap, bw, bh, "DESCER", ("rel", "z_down",), DIR))

        # passos (linha embaixo)
        sy = 470
        for i, s in enumerate([1, 5, 10]):
            color = (0, 130, 0) if s == self.step else UTIL
            self.buttons.append(Button(40 + i * 90, sy, 80, 40, f"{s} mm", ("step", s), color))

        # feed +/-
        self.buttons.append(Button(330, sy, 60, 40, "F-", ("feed", -500), UTIL))
        self.buttons.append(Button(395, sy, 60, 40, "F+", ("feed", +500), UTIL))

        # acoes utilitarias
        self.buttons.append(Button(490, sy, 100, 40, "HOME", ("home",), BIG))
        self.buttons.append(Button(600, sy, 130, 40, "MOTORS OFF", ("motors_off",), UTIL))

        # bottom row
        sy2 = 530
        self.buttons.append(Button(40, sy2, 130, 50, "READ POS", ("read_pos",), UTIL))
        self.buttons.append(Button(180, sy2, 130, 50, "Z=200", ("goto", 0, 0, 200), UTIL))
        self.buttons.append(Button(320, sy2, 130, 50, "Z=290", ("goto", 0, 0, 290), UTIL))
        self.buttons.append(Button(610, sy2, 120, 50, "QUIT", ("quit",), DANGER))

    def on_mouse(self, event, x, y, flags, param):
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def render(self):
        img = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        # title
        cv2.putText(img, "DELTA JOG", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)

        # pos
        if self.pos:
            x, y, z = self.pos
            r = (x * x + y * y) ** 0.5
            line1 = f"X={x:7.2f}   Y={y:7.2f}   Z={z:7.2f}"
            line2 = f"raio={r:6.2f}     passo={self.step}mm     feed={self.feed}"
        else:
            line1 = "Posicao desconhecida"
            line2 = "Aperte HOME pra fazer homing"

        cv2.putText(img, line1, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, line2, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # guards
        gline = f"guards: raio<={self.max_radius}   {self.z_min}<=Z<={self.z_max}"
        cv2.putText(img, gline, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        # status
        if self.busy:
            cv2.putText(img, "...mexendo...", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            cv2.putText(img, self.msg[:80], (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        # update step button highlight
        for b in self.buttons:
            if isinstance(b.action, tuple) and b.action[0] == "step":
                b.color = (0, 130, 0) if b.action[1] == self.step else (100, 60, 100)

        # hover
        mx, my = self.mouse
        for b in self.buttons:
            b.hover = b.hit(mx, my)
            b.draw(img)

        # rodape
        foot = "Tambem aceita teclado: WASD RF 1/2/3 +/- H SPACE M Q"
        cv2.putText(img, foot, (20, WIN_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)

        return img

    def do_action(self, action):
        if action[0] == "rel":
            # acoes possiveis:
            #   ("rel", "x", sign), ("rel", "y", sign): movimento horizontal
            #   ("rel", "z_up",): efetuador SOBE fisicamente
            #   ("rel", "z_down",): efetuador DESCE fisicamente
            tag = action[1]
            if tag in ("x", "y"):
                _, axis, sign = action
                self._move_rel(axis, sign * self.step)
            elif tag == "z_up":
                # subir: se invert_z True, Z aumenta no software pra subir
                # caso contrario (convencao Marlin padrao), Z aumenta pra subir
                # (na nossa firmware sprinter, invert_z=True significa que
                # Z aumenta indo pra baixo -> entao pra subir, decrementa)
                delta = -self.step if self.invert_z else +self.step
                self._move_rel("z", delta)
            elif tag == "z_down":
                delta = +self.step if self.invert_z else -self.step
                self._move_rel("z", delta)
        elif action[0] == "step":
            self.step = action[1]
            self.msg = f"passo = {self.step}mm"
        elif action[0] == "feed":
            self.feed = max(500, min(self.feed + action[1], 6000))
            self.msg = f"feed = {self.feed}"
        elif action[0] == "home":
            self._home()
        elif action[0] == "motors_off":
            send(self.ser, "M84", wait=1)
            self.pos = None
            self.msg = "Motores desligados — pode mexer na mao"
        elif action[0] == "read_pos":
            self.pos = get_position(self.ser)
            self.msg = "Posicao atualizada"
        elif action[0] == "goto":
            _, gx, gy, gz = action
            self._goto(gx, gy, gz)
        elif action[0] == "quit":
            self._quit()

    def _check_guards(self, x, y, z):
        if (x * x + y * y) ** 0.5 > self.max_radius:
            return f"BLOQUEADO: raio > {self.max_radius}mm"
        if z < self.z_min:
            return f"BLOQUEADO: Z < {self.z_min}mm"
        if z > self.z_max:
            return f"BLOQUEADO: Z > {self.z_max}mm"
        return None

    def _move_rel(self, axis, delta):
        if self.pos is None:
            self.msg = "Sem posicao — aperte HOME primeiro"
            return
        x, y, z = self.pos
        if axis == "x": x += delta
        elif axis == "y": y += delta
        else: z += delta
        err = self._check_guards(x, y, z)
        if err:
            self.msg = err
            return
        self.busy = True
        send(self.ser, f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{self.feed}", wait=2)
        send(self.ser, "M400", wait=5)
        self.pos = (x, y, z)
        self.msg = f"-> ({x:.1f}, {y:.1f}, {z:.1f})"
        self.busy = False

    def _goto(self, x, y, z):
        err = self._check_guards(x, y, z)
        if err:
            self.msg = err
            return
        self.busy = True
        send(self.ser, f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{self.feed}", wait=2)
        send(self.ser, "M400", wait=10)
        self.pos = (x, y, z)
        self.msg = f"-> ({x:.0f}, {y:.0f}, {z:.0f})"
        self.busy = False

    def _home(self):
        self.busy = True
        self.msg = "Homing..."
        send(self.ser, "G28", wait=30, timeout_extra=30)
        send(self.ser, "M400", wait=10)
        self.pos = get_position(self.ser)
        self.msg = "Home OK"
        self.busy = False

    def _quit(self):
        self.busy = True
        self.msg = "Subindo para Z=290 e saindo..."
        send(self.ser, f"G1 X0 Y0 Z290 F{self.feed}", wait=10)
        send(self.ser, "M400", wait=10)
        send(self.ser, "M84", wait=1)
        self.quit = True

    def handle_key(self, key):
        if key == 255 or key < 0:
            return
        ch = chr(key) if 0 <= key < 128 else None
        mapping = {
            "q": ("quit",),
            "h": ("home",),
            " ": ("read_pos",),
            "m": ("motors_off",),
            "1": ("step", 1),
            "2": ("step", 5),
            "3": ("step", 10),
            "+": ("feed", 500),
            "=": ("feed", 500),
            "-": ("feed", -500),
            "d": ("rel", "x", +1),
            "a": ("rel", "x", -1),
            "w": ("rel", "y", +1),
            "s": ("rel", "y", -1),
            "r": ("rel", "z_up",),
            "f": ("rel", "z_down",),
        }
        action = mapping.get(ch)
        if action:
            self.do_action(action)

    def loop(self):
        win = "Delta Jog"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, WIN_W, WIN_H)
        cv2.setMouseCallback(win, self.on_mouse)

        while not self.quit:
            cv2.imshow(win, self.render())
            key = cv2.waitKey(30) & 0xFF
            if key != 255:
                self.handle_key(key)

            if self.click is not None:
                mx, my = self.click
                self.click = None
                for b in self.buttons:
                    if b.hit(mx, my):
                        b.pressed_until = time.time() + 0.15
                        self.do_action(b.action)
                        break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--feed", type=int, default=2000)
    parser.add_argument("--max-radius", type=float, default=55)
    parser.add_argument("--z-min", type=float, default=50)
    parser.add_argument("--z-max", type=float, default=300)
    parser.add_argument("--no-home", action="store_true")
    parser.add_argument("--invert-z", action="store_true",
                        help="Inverte direcao de SUBIR/DESCER (firmware Sprinter antigo)")
    args = parser.parse_args()

    print(f"Conectando em {args.port} ...")
    ser = serial.Serial(args.port, args.baud, timeout=2)
    time.sleep(3)
    ser.reset_input_buffer()
    send(ser, "G21", wait=1)
    send(ser, "G90", wait=1)

    app = JogApp(ser, args)

    if not args.no_home:
        print("Fazendo homing inicial ...")
        app._home()
    try:
        app.loop()
    except KeyboardInterrupt:
        print("\nCancelado.")
    finally:
        ser.close()
        print("Conexao fechada.")


if __name__ == "__main__":
    main()
