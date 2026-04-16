# Setup no Notebook - Gravar SD e Configurar RPi

Abra o terminal (PowerShell ou Windows Terminal) e siga os passos abaixo.

## 1. Instalar o Raspberry Pi Imager

```bash
winget install RaspberryPiFoundation.RaspberryPiImager
```

Se nao tiver winget, baixe direto: https://www.raspberrypi.com/software/

## 2. Espete o microSD no notebook

Use o slot SD do notebook (com adaptador microSD -> SD se necessario).

## 3. Abra o Raspberry Pi Imager

Procure "Raspberry Pi Imager" no menu iniciar e abra.

### 3.1 Selecionar dispositivo
- Clique em **Dispositivo**
- Selecione **Raspberry Pi 4**

### 3.2 Selecionar sistema operacional
- Clique em **Sistema Operacional**
- Selecione **Raspberry Pi OS (64-bit)**
- **TEM QUE SER 64-BIT** (senao o PyTorch nao funciona)

### 3.3 Selecionar armazenamento
- Clique em **Armazenamento**
- Selecione o seu **microSD** (cuidado pra nao selecionar outro disco!)

### 3.4 Configurar antes de gravar
- Clique em **Proximo**
- Vai aparecer "Gostaria de aplicar as configuracoes de personalizacao do SO?"
- Clique em **Editar configuracoes**

#### Aba GERAL:
| Campo | Valor |
|-------|-------|
| Hostname | `moneypi` |
| Nome de usuario | `pi` |
| Senha | escolha uma e **ANOTE** |
| SSID do WiFi | nome da sua rede WiFi |
| Senha do WiFi | senha da sua rede |
| Pais WiFi | `BR` |
| Fuso horario | `America/Sao_Paulo` |
| Layout teclado | `br` |

#### Aba SERVICOS:
| Campo | Valor |
|-------|-------|
| Habilitar SSH | **SIM** |
| Autenticacao | Usar senha |

- Clique **Salvar**
- Clique **Sim** para aplicar
- Clique **Sim** para confirmar que vai apagar o SD
- **Espere gravar e verificar** (~5-10 minutos, nao remova o SD!)

### 3.5 Finalizar
- Quando aparecer "Gravacao concluida", ejete o SD com seguranca
- Remova o microSD do notebook

## 4. Montar o Raspberry Pi

1. Insira o microSD no RPi4 (slot embaixo da placa)
2. Conecte o cabo flat da camera no conector CSI
   - Levante a trava plastica do conector
   - Encaixe o cabo com o lado azul virado pro slot do SD
   - Abaixe a trava
3. Conecte cabo ethernet (opcional se configurou WiFi)
4. Conecte a fonte USB-C -> RPi liga automaticamente
5. **Espere 2-3 minutos** (primeiro boot e mais demorado)

## 5. Conectar no RPi via SSH

De volta no terminal do notebook:

```bash
ssh pi@moneypi.local
```

- Se pedir "Are you sure you want to continue connecting?" -> digite `yes`
- Digite a senha que voce configurou no passo 3.4

**Se `moneypi.local` nao funcionar:**
- Entre no seu roteador (geralmente 192.168.1.1) e procure o IP do "moneypi"
- Ou rode: `ping moneypi.local` pra ver se resolve
- Tente: `ssh pi@<IP_DO_RPI>`

## 6. Instalar tudo no RPi (via SSH)

Agora voce esta dentro do RPi. Cole esses comandos:

```bash
# Clona o repositorio
cd ~
git clone https://github.com/gabrielDallape/money_pick_n_place.git
cd money_pick_n_place

# Roda o setup automatico (demora ~15-30 min, deixe rodar)
bash setup_rpi.sh
```

## 7. Baixar o modelo treinado

```bash
source ~/money_env/bin/activate
cd ~/money_pick_n_place

# Opcao A: via GitHub CLI (se tiver gh instalado)
sudo apt install -y gh
gh auth login
gh release download v1.0 --pattern "best.pt" --dir .

# Opcao B: via curl direto
curl -L -o best.pt https://github.com/gabrielDallape/money_pick_n_place/releases/download/v1.0/best.pt
```

## 8. Testar a camera

```bash
# Verifica se camera e detectada
libcamera-still --list-cameras

# Tira uma foto de teste
libcamera-still -o teste_camera.jpg
```

Se nao detectar a camera:
```bash
sudo raspi-config
# -> Interface Options -> Camera -> Enable
# -> Finish -> Reboot
# Depois reconecte via SSH e tente de novo
```

## 9. Rodar a deteccao

```bash
source ~/money_env/bin/activate
cd ~/money_pick_n_place

# Captura unica: tira foto e detecta
python rpi_detect.py --model best.pt

# Loop a cada 5 segundos
python rpi_detect.py --model best.pt --loop --interval 5

# Se tiver monitor/VNC: preview ao vivo
python rpi_detect.py --model best.pt --live
```

### Saida esperada:
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

## Resumo dos comandos (copia/cola rapido)

```bash
# No notebook:
ssh pi@moneypi.local

# No RPi (primeira vez):
cd ~ && git clone https://github.com/gabrielDallape/money_pick_n_place.git && cd money_pick_n_place && bash setup_rpi.sh

# No RPi (depois do setup):
source ~/money_env/bin/activate && cd ~/money_pick_n_place
curl -L -o best.pt https://github.com/gabrielDallape/money_pick_n_place/releases/download/v1.0/best.pt
python rpi_detect.py --model best.pt
```
