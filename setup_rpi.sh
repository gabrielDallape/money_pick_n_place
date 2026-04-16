#!/bin/bash
# Setup do Raspberry Pi 4 para deteccao de cedulas com YOLOv8
# Rodar no RPi: bash setup_rpi.sh

set -e

echo "=== Setup Money Pick and Place - RPi4 ==="

# Atualiza sistema
sudo apt update && sudo apt upgrade -y

# Dependencias do OpenCV e camera
sudo apt install -y python3-pip python3-venv libcamera-apps \
    libatlas-base-dev libhdf5-dev libopenblas-dev libjpeg-dev \
    libpng-dev libtiff-dev libavcodec-dev libavformat-dev \
    libswscale-dev libv4l-dev v4l-utils

# Cria venv
python3 -m venv ~/money_env
source ~/money_env/bin/activate

# Instala ultralytics (inclui opencv-python, torch, etc)
pip install --upgrade pip
pip install ultralytics

# Testa se camera funciona
echo ""
echo "=== Testando camera ==="
if [ -e /dev/video0 ]; then
    echo "Camera encontrada em /dev/video0"
else
    echo "AVISO: /dev/video0 nao encontrado."
    echo "Verifique:"
    echo "  1. sudo raspi-config -> Interface Options -> Camera -> Enable"
    echo "  2. Adicione 'start_x=1' no /boot/config.txt"
    echo "  3. Reinicie o RPi"
fi

echo ""
echo "=== Setup concluido! ==="
echo "Para usar:"
echo "  source ~/money_env/bin/activate"
echo "  python rpi_detect.py --model best.pt"
