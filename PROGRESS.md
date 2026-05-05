# Money Pick & Place — Progresso e Próximos Passos

> Última atualização: **2026-05-05**

Projeto: usar visão computacional pra detectar cédulas de Real numa cena e guiar uma impressora 3D Delta (Puzzles 3D) convertida em pick & place pra pegar e organizar as notas.

---

## Estado atual

### ✅ Visão computacional

| Item | Status | Detalhes |
|---|---|---|
| Detector v1 | ✅ baseline | YOLOv8n, mAP@50 = 0.964, mAP@50-95 = 0.864 |
| Dataset v2 (Roboflow) | ✅ baixado | 7451 imagens, 7 classes (R$2/5/10/20/50/100/200) |
| Split sem leakage | ✅ | `prepare_v2_dataset.py` agrupa por foto-fonte (rotações 0/90/180/270 ficam no mesmo split). Resultado: 6091 train / 973 valid / 387 test / 50 holdout |
| Holdout congelado | ✅ | 50 imgs, SHA256, lock em `HOLDOUT_LOCK.md` |
| Detector v2 | ✅ treinado | mAP@50 = 0.948, mAP@50-95 = 0.805. Pesos: `runs/detect/runs/detect/v2_20260504_172303/weights/best.pt`. Transfer learning a partir do v1, parou em 27 épocas (early stopping). |
| Versionamento DVC | ✅ | `datasets/v2/` rastreado, push pro storage local em `~/Desktop/dvc_storage_money_pnp` |
| W&B tracking | ✅ | runs em `wandb.ai/gadallape-gabriel-dallape/money-pick-n-place` |
| Capturador de fotos | ✅ | `capture_dataset.py` com auto-mode, blur filter, organização por sessão |
| Realtime test | ✅ | `realtime_test.py` — bbox arredondada + centro + detecção de dobrada + ponto ótimo de pega via distance transform |
| Avaliação detalhada | ✅ script | `evaluate_model.py` (matriz confusão + galeria erros) |
| Inspeção visual | ✅ script | `inspect_dataset.py` (FiftyOne UI no browser) |
| Slice metrics | ✅ script | `slice_metrics.py` (F1 por brilho/blur/rotação/contagem) |
| Bad labels detection | ✅ script | `find_bad_labels.py` (rank de imagens provavelmente mal-rotuladas) |
| Active learning | ✅ script | `active_learn.py` (rank de incerteza pra próxima rodada) |
| Model card template | ✅ | `MODEL_CARD.md` (padrão Google) |
| Dockerfile reproduzível | ✅ | `Dockerfile` + `.dockerignore` |

### ✅ Hardware Delta (Puzzles 3D)

| Item | Status | Detalhes |
|---|---|---|
| Comunicação serial | ✅ | CH340 em COM3/COM4, 115200 baud, Marlin V1 (Sprinter mashup) |
| Homing | ✅ | G28 funciona, Z máximo = 305mm |
| Movimento básico | ✅ | G0/G1 X/Y/Z, cinemática delta correta (counts diferentes nos 3 carros) |
| Raio máximo medido | ✅ | **60mm** (área útil = círculo de 12cm de diâmetro) |
| Z mínimo (mesa) | ❌ | não medido ainda |
| Jog manual | ✅ | `delta_jog.py` (janela OpenCV com botões) |
| Quirk firmware | ⚠️ | "snap-back" de 0.009mm no primeiro G1 após G28. Inofensivo (motor zumbe, não há motion real). Para resolver definitivo: reflashar Marlin 2.x. |

### ❌ Integração CV ↔ Delta

Ainda não começou. Aguardando hardware (webcam montada, base, efetuador).

---

## Próximos passos

### Fase 1 — Visão computacional (ATUAL FOCO)

Antes de mexer no hardware, queremos finalizar a parte de CV:

- [ ] **Validar v2 na webcam** com `realtime_test.py` em condições reais (vários fundos, luzes, dobras). Comparar com v1 anedotalmente.
- [ ] **Rodar `evaluate_model.py`** no split `valid` pra extrair métricas por classe (e identificar qual cédula está mais fraca).
- [ ] **Rodar `slice_metrics.py`** pra ver em que condições (brilho, blur, count) o modelo é fraco. Direciona o que coletar/rotular na v3.
- [ ] **Rodar `find_bad_labels.py`** pra achar labels suspeitos no Roboflow e re-rotular.
- [ ] **Rodar `inspect_dataset.py`** pra navegar o dataset visualmente no FiftyOne. Marcar amostras estranhas com tag `review`.
- [ ] (Se necessário) **Capturar mais fotos** com `capture_dataset.py` em cenários ruins identificados (`--tag dobradas`, `--tag fundo_escuro`, `--tag negativos`...). Subir no Roboflow → re-treinar v3.
- [ ] **Atualizar `MODEL_CARD.md`** com as métricas reais do v2.

### Fase 2 — Hardware (PRÓXIMO BLOCO, depois da CV estar boa)

Itens a fabricar/comprar:

#### 2.1 Suporte de webcam
Câmera precisa ficar **rigidamente fixa** acima da área de trabalho da Delta, olhando pra baixo. Qualquer vibração na câmera = erro de detecção em mm no robô.

- Material: peça impressa em PETG/PLA pra prender no perfil de alumínio da Delta ou num braço externo
- Altura ideal: ~30-50cm acima da base de trabalho (suficiente pra ver toda a área de 12cm de raio com folga)
- Lente apontada pro centro do círculo de trabalho

#### 2.2 Base pra colocar o dinheiro
Superfície plana, com cor uniforme (preto fosco ou branco), do tamanho da área útil + margem.

- Tamanho mínimo: 15cm x 15cm
- **4 marcadores ArUco impressos nos cantos** pra calibração câmera↔mundo (vamos detalhar quando chegar a hora)
- Material: papel cartão preto fosco ou MDF pintado
- Anti-reflexo (importante pra evitar problema de iluminação no detector)

#### 2.3 Ponteira que rotaciona + reuso do tubo do outro projeto
Efetuador (ferramenta na ponta da Delta) com:

- **Ventosa de silicone** (Φ ~10-15mm, suficiente pra cédula ~140x65mm)
- **Servo-motor** pra rotacionar a ponta — porque cédulas vêm em ângulos variados, e pegar elas alinhadas facilita organizar depois
- **Tubo do outro projeto** reaproveitado pra conduzir o vácuo da bombinha até a ventosa
- Bombinha de vácuo 12V (ex: bomba de aquário modificada ou peça vendida pra pick-and-place SMD)
- Acionamento da bomba via porta GPIO da placa Marlin (M42 P<pino> S0/S255)

### Fase 3 — Integração

- [ ] Calibração câmera↔mundo com **homografia ArUco** (~30 min de código + tuning)
- [ ] Loop completo:
  1. Webcam captura frame
  2. YOLO detecta cédulas + centros + dobradas + ponto ótimo de pega
  3. Pixel→mm via homografia
  4. G-code envia Delta pra (X, Y, Z=alto)
  5. Desce até Z baixo
  6. Liga vácuo (M42)
  7. Sobe
  8. Move pra zona de descarte/organização
  9. Desliga vácuo
  10. Repete
- [ ] Testes de robustez (várias notas, dobradas, sobrepostas)
- [ ] Otimização de ordem de pega (qual nota pegar primeiro pra evitar que outras se mexam)

---

## Arquivos importantes do projeto

### Scripts de visão computacional
- `train_v2.py` — treino com transfer learning, W&B, seeds determinísticos
- `prepare_v2_dataset.py` — split 80/15/5 sem leakage entre rotações
- `freeze_holdout.py` — congela golden set com SHA256
- `evaluate_model.py` — métricas + galeria de erros
- `inspect_dataset.py` — UI FiftyOne com predições sobrepostas
- `find_bad_labels.py` — rank de labels suspeitos
- `active_learn.py` — rank de incerteza pra anotação
- `slice_metrics.py` — F1 por condição (brilho/blur/rotação/contagem)
- `realtime_test.py` — webcam ao vivo com detecção + dobra + ponto de pega
- `capture_dataset.py` — captura novas fotos (auto/manual, com blur filter)

### Scripts de hardware (Delta)
- `delta_hello.py` — primeiro contato (firmware info, M114, opcional G28)
- `delta_test_move.py` — teste de movimento controlado
- `delta_find_radius.py` — descobre raio máximo seguro
- `delta_jog.py` — controle manual via janela OpenCV (botões clicáveis)
- `delta_diagnose_home.py` — diagnóstico de comportamento de homing

### Documentação
- `README.md` — visão geral original do projeto
- `MODEL_CARD.md` — model card no padrão Google (template)
- `HOLDOUT_LOCK.md` — auditoria do golden set
- `PROGRESS.md` — esse arquivo
- `SETUP_NOTEBOOK.md` — guia passo-a-passo original

### Infra
- `Dockerfile` + `.dockerignore` — imagem reproduzível
- `requirements.txt` — deps incluindo wandb, fiftyone, cleanlab
- `.dvc/` — config DVC (dataset versionado em remote local)

---

## Como começar do zero (próximo dev / outra máquina)

```bash
# 1. clonar
git clone https://github.com/gabrielDallape/money_pick_n_place.git
cd money_pick_n_place

# 2. instalar deps
pip install -r requirements.txt

# 3. baixar dataset versionado (precisa do storage local ou cloud configurado)
dvc pull

# 4. treinar v2 (parte do v1 best.pt)
wandb login
python train_v2.py

# 5. avaliar
python evaluate_model.py
python slice_metrics.py

# 6. testar na webcam
python realtime_test.py --model runs/detect/runs/detect/v2_*/weights/best.pt

# 7. (quando tiver hardware) testar Delta
python delta_hello.py --port COM3
python delta_jog.py --port COM3
```

---

## Decisões tomadas / lições aprendidas

- **Marlin V1 antigo (Sprinter mashup) tem quirks** de motion control. Ignorar comportamentos inofensivos (snap-back de 0.009mm) e não tentar burlar com truques tipo `G92` — pode confundir a contabilidade interna e causar movimentos descontrolados.
- **Sempre ter uma trava de raio em software** no controle da Delta. O firmware antigo nem sempre rejeita movimentos fora de limite — pode silenciosamente clipar ou travar.
- **Split de dataset por foto-fonte** (não por imagem individual) é essencial quando o dataset tem múltiplas rotações/augmentações da mesma cena. Senão = data leakage.
- **Holdout congelado** é não-negociável pra ter métrica final honesta. O test set normal contamina ao longo da iteração.
- **Métricas agregadas mentem.** Slice metrics revela onde o modelo é fraco de verdade.
- **Câmera precisa ser rigidamente fixa.** 1mm de vibração na câmera = ~1cm de erro no robô (depende da altura).
