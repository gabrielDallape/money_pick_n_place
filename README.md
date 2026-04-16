# Money Pick and Place

Sistema robotico pick-and-place para identificar, coletar e organizar cedulas brasileiras usando visao computacional (YOLOv8) + succao a vacuo.

## Arquitetura

- **Raspberry Pi 4**: visao (camera + YOLO), decisao, geracao de G-code
- **Impressora 3D Delta Creality**: movimentacao via firmware Marlin (G-code via USB serial)
- **Camera fixa top-down**: modulo camera RPi
- **Ventosa com succao a vacuo**: controlada via fan output (M106/M107)

## Estrutura do projeto

```
money_pick_n_place/
├── train.py              # Treino YOLOv8 para deteccao de cedulas
├── detect.py             # Deteccao via webcam/imagem (PC)
├── rpi_detect.py         # Deteccao no Raspberry Pi 4 (camera module)
├── download_dataset.py   # Download dataset do Roboflow
├── setup_rpi.sh          # Setup automatico do RPi
├── progress.py           # Barra de progresso do treino
├── requirements.txt      # Dependencias Python
└── datasets/             # (gitignored - baixar via download_dataset.py)
```

## Modelo treinado

O modelo `best.pt` (YOLOv8m, 50 MB) esta disponivel na aba **Releases** deste repositorio.

**Metricas do treino (150 epocas):**
- mAP50: **0.810**
- mAP50-95: **0.495**
- Precision: **0.876**
- Recall: **0.752**

Classes detectadas: `100back`, `100front`, `100true`, `200back`, `200front`, `50back`, `50front`, `50true`

---

# Guia: Setup do Raspberry Pi 4 (do zero)

Guia completo para configurar o RPi4 com camera e rodar a deteccao de cedulas.

## O que voce precisa

- Raspberry Pi 4 (2GB+ RAM)
- Modulo de camera RPi (v1, v2 ou v3)
- MicroSD 16GB+ (recomendo 32GB)
- Fonte USB-C 5V/3A para o RPi
- Cabo ethernet OU WiFi configurado
- No notebook: leitor de cartao SD

## Passo 1 — Gravar o Raspberry Pi OS no microSD

### 1.1 Instalar o Raspberry Pi Imager

Baixe em: https://www.raspberrypi.com/software/

Ou no terminal (Windows):
```
winget install RaspberryPiFoundation.RaspberryPiImager
```

### 1.2 Configurar e gravar

1. Abra o **Raspberry Pi Imager**
2. Clique em **Dispositivo** -> selecione **Raspberry Pi 4**
3. Clique em **Sistema Operacional** -> selecione **Raspberry Pi OS (64-bit)**
   - IMPORTANTE: tem que ser **64-bit** para o PyTorch funcionar
4. Clique em **Armazenamento** -> selecione seu **microSD**
5. Clique em **Proximo** -> vai aparecer "Gostaria de personalizar?" -> clique **Editar configuracoes**

### 1.3 Configuracoes importantes (tela de engrenagem)

Na aba **GERAL**:
- Hostname: `moneypi`
- Nome de usuario: `pi`
- Senha: (escolha uma senha, anote!)
- WiFi: coloque o SSID e senha da sua rede
- Pais WiFi: `BR`
- Fuso horario: `America/Sao_Paulo`
- Layout do teclado: `br`

Na aba **SERVICOS**:
- Habilitar SSH: **SIM** (usar autenticacao por senha)

6. Clique **Salvar** -> **Sim** -> **Sim** (confirma apagar o SD)
7. Espere gravar e verificar (~5-10 minutos)
8. Ejete o cartao com seguranca

## Passo 2 — Primeiro boot do RPi

1. Insira o microSD no Raspberry Pi 4
2. Conecte o cabo da camera no conector CSI (a aba azul fica virada pro slot de SD)
3. Conecte o cabo ethernet (opcional se configurou WiFi)
4. Conecte a fonte USB-C — o RPi vai ligar automaticamente
5. Espere ~2 minutos pro primeiro boot (expande o filesystem, configura etc.)

## Passo 3 — Conectar via SSH pelo notebook

### 3.1 Descobrir o IP do RPi

Tente primeiro pelo hostname:
```bash
ssh pi@moneypi.local
```

Se nao funcionar, descubra o IP:
- No roteador: veja a lista de dispositivos conectados, procure "moneypi"
- Ou escaneie a rede: `nmap -sn 192.168.1.0/24` (ou o range da sua rede)

### 3.2 Primeira conexao SSH

```bash
ssh pi@moneypi.local
# Aceite a fingerprint (yes)
# Digite a senha que configurou
```

Se estiver no Windows sem SSH nativo, use o **Windows Terminal** ou **PuTTY**.

## Passo 4 — Habilitar a camera no RPi

```bash
# Verifica se a camera e detectada
libcamera-still --list-cameras

# Se aparecer uma camera, teste tirar uma foto:
libcamera-still -o teste.jpg
```

Se nao detectar:
```bash
# Verifica se o cabo esta bem conectado
# Verifica se a camera esta habilitada:
sudo raspi-config
# -> Interface Options -> Camera -> Enable
# -> Reboot
```

## Passo 5 — Clonar o repositorio e instalar dependencias

```bash
# No RPi via SSH:
cd ~

# Clona o repositorio
git clone https://github.com/gabrielDallape/money_pick_n_place.git
cd money_pick_n_place

# Roda o script de setup (instala tudo)
bash setup_rpi.sh
```

O `setup_rpi.sh` faz:
- Atualiza o sistema
- Instala dependencias (OpenCV, libcamera, etc.)
- Cria um ambiente virtual Python em `~/money_env`
- Instala ultralytics + PyTorch

**ATENCAO:** Esse passo demora ~15-30 minutos no RPi4. Deixe rodar.

## Passo 6 — Baixar o modelo treinado

```bash
# Ainda no RPi via SSH:
source ~/money_env/bin/activate
cd ~/money_pick_n_place

# Baixa o best.pt da Release do GitHub
gh release download v1.0 --pattern "best.pt" --dir .

# OU baixe manualmente pelo navegador na aba Releases do repositorio
# OU transfira direto do PC que treinou:
```

**Alternativa — transferir direto do PC via SCP:**
```bash
# Rodar isso no PC (NAO no RPi), substitua <IP_RPI> pelo IP do RPi:
scp C:/Users/gadal/Desktop/money_pick_n_place/runs/detect/runs/cedulas_gpu/weights/best.pt pi@<IP_RPI>:~/money_pick_n_place/
```

## Passo 7 — Testar a deteccao

```bash
# No RPi:
source ~/money_env/bin/activate
cd ~/money_pick_n_place

# Teste 1: captura unica (tira foto, detecta, salva resultado)
python rpi_detect.py --model best.pt

# Teste 2: detectar em loop a cada 5 segundos
python rpi_detect.py --model best.pt --loop --interval 5

# Teste 3: preview ao vivo (precisa de monitor conectado ou VNC)
python rpi_detect.py --model best.pt --live

# Teste 4: detectar em imagem existente
python rpi_detect.py --model best.pt --image teste.jpg
```

### Saida esperada

```
Carregando modelo: best.pt
Modelo carregado. Classes: ['100back', '100front', ...]
Capturando foto...

--- 2 cedula(s) detectada(s) [3.45s] ---
    50front  conf=0.87  centro=(320, 240)
   100back  conf=0.72  centro=(150, 180)

Salvo: captura_original.jpg, captura_detectada.jpg
Deteccoes salvas em: deteccoes.json
```

O arquivo `deteccoes.json` contem as coordenadas pro sistema pick-and-place:
```json
[
  {
    "class": "50front",
    "confidence": 0.87,
    "bbox": [200.0, 150.0, 440.0, 330.0],
    "center": [320.0, 240.0]
  }
]
```

## Modos do rpi_detect.py

| Flag | Descricao |
|------|-----------|
| (nenhuma) | Captura unica: tira 1 foto, detecta, salva |
| `--live` | Preview ao vivo com FPS (monitor/VNC) |
| `--loop` | Loop continuo (default: 5s entre capturas) |
| `--interval N` | Muda intervalo do loop para N segundos |
| `--image X` | Detecta numa imagem existente |
| `--conf 0.5` | Confianca minima (0 a 1, default 0.5) |

## Troubleshooting

### Camera nao detectada
```bash
# Verifica conexao
vcgencmd get_camera    # deve mostrar: detected=1
# OU no Bookworm:
libcamera-still --list-cameras
```

### Erro de memoria ao carregar modelo
O YOLOv8m usa ~500MB RAM. Se o RPi4 tem 2GB, feche tudo antes:
```bash
# Mata processos desnecessarios
sudo systemctl stop lightdm   # desliga interface grafica
```

### Muito lento (>10s por frame)
Exporte o modelo para NCNN (otimizado pra ARM):
```bash
source ~/money_env/bin/activate
python -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='ncnn')
"
# Depois use: python rpi_detect.py --model best_ncnn_model
```

### SSH nao conecta
- Verifique se RPi e notebook estao na mesma rede
- Tente pelo IP direto em vez de `moneypi.local`
- Verifique se SSH foi habilitado no Imager
