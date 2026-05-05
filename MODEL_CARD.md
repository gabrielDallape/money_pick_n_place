# Model Card — Money Pick & Place (Detector de Cédulas BRL)

> Template no padrão Google Model Cards. Atualize as seções marcadas com **[PREENCHER]** após cada release de modelo. Comite junto do `best.pt` no git/W&B.

## Detalhes do Modelo

- **Nome:** money-pnp-yolov8
- **Versão:** v2.0 **[PREENCHER após treino]**
- **Data:** **[PREENCHER]**
- **Tipo:** YOLOv8n (object detection, 7 classes)
- **Treinado por:** Gabriel Dallape (gabriel.dallape@norfolk.ai)
- **Repositório:** **[link do GitHub]**
- **Commit do treino:** **[git rev-parse HEAD]**
- **Pesos:** `runs/detect/v2_<timestamp>/weights/best.pt`
- **Pesos iniciais (transfer learning):** `runs/detect/runs/cedulas2/weights/best.pt` (v1)

## Uso Pretendido

- **Aplicação:** detectar e classificar cédulas de Real (BRL) numa cena, em tempo real, para guiar um sistema de pick & place (originalmente Raspberry Pi 4 com webcam).
- **Usuários alvo:** o próprio autor (projeto pessoal) e qualquer pessoa interessada em CV aplicado a manipulação robótica de objetos.
- **Casos de uso ESPERADOS:**
  - Detecção em ambiente controlado (mesa lisa, iluminação consistente)
  - Notas separadas ou levemente sobrepostas
  - Notas levemente dobradas (heurística de fold detection no `realtime_test.py`)
- **Casos de uso NÃO suportados:**
  - Verificação de autenticidade (NÃO detecta nota falsa)
  - Contagem precisa de valor monetário em cenas com muita oclusão
  - Notas estrangeiras
  - Iluminação extrema (escuro, luz direta na lente)

## Classes

| ID | Nome | Denominação |
|---|---|---|
| 0 | `10` | R$ 10 |
| 1 | `100` | R$ 100 |
| 2 | `2` | R$ 2 |
| 3 | `20` | R$ 20 |
| 4 | `200` | R$ 200 |
| 5 | `5` | R$ 5 |
| 6 | `50` | R$ 50 |

## Dados de Treino

- **Origem:** Roboflow workspace `detectorobjetos`, projeto `pick-n-place-money`, versão v2 (exportada em 2026-05-03)
- **Total de imagens:** 7451
- **Splits (após `prepare_v2_dataset.py`, seed=42):**
  - train: 6091
  - valid: 973
  - test: 337
  - holdout (golden, congelado): 50 — ver `HOLDOUT_LOCK.md`
- **Anti-leakage:** split agrupado por foto-fonte (rotações 0/90/180/270 do mesmo original ficam no mesmo split)
- **Preprocessing:** nenhum aplicado pelo Roboflow nessa versão
- **Augmentation no treino:** HSV±0.4, rotação ±15°, scale 0.5, mosaic, mixup 0.1, erasing 0.4
- **Limitações conhecidas dos dados:**
  - Variabilidade de fundo limitada → modelo v1 detectava no chão (ver seção Limitações)
  - Distribuição desbalanceada entre classes (R$100 historicamente fraca no v1)
  - Pouca representação de notas dobradas/amassadas

## Avaliação

> Preencher após `python evaluate_model.py` e `python slice_metrics.py`

### Métricas no split `valid` (early stopping)

| Métrica | Valor |
|---|---|
| mAP@50 | **[PREENCHER]** |
| mAP@50-95 | **[PREENCHER]** |
| Precision | **[PREENCHER]** |
| Recall | **[PREENCHER]** |

### Métricas por classe

| Classe | Precision | Recall | mAP@50 |
|---|---|---|---|
| R$2 | | | |
| R$5 | | | |
| R$10 | | | |
| R$20 | | | |
| R$50 | | | |
| R$100 | | | |
| R$200 | | | |

### Slice metrics (do `slice_metrics.py`)

- **Pior fatia por F1:** **[PREENCHER]**
- **Melhor fatia por F1:** **[PREENCHER]**
- Detalhes em `runs/slices/<run>/slices.md`

### Holdout final (golden set, abrir só na release)

- **Data da abertura:** **[PREENCHER]**
- **mAP@50 no holdout:** **[PREENCHER]**
- **mAP@50-95 no holdout:** **[PREENCHER]**
- **Diferença vs valid:** **[PREENCHER]** (gap > 5pp pode indicar overfit)

## Considerações Éticas / Viéses Conhecidos

- **Variação de iluminação:** dataset capturado predominantemente sob luz artificial — desempenho pode degradar sob luz solar direta.
- **Ângulo de câmera:** otimizado para ~0-30° de elevação (top-down e oblíquo). Vista lateral não testada.
- **Cor de fundo:** dataset com variabilidade limitada — usuário deve testar no fundo do seu cenário antes de confiar.
- **Cédulas antigas vs novas:** versão atual da família real cobre principalmente notas pós-2010.

## Limitações

1. **Não detecta notas falsas.** Só classifica denominação aparente.
2. **Confunde denominações em condições ruins.** R$10 e R$100 podem ser confundidos com baixa luz.
3. **Detecções no fundo (problema do v1).** Em fundos texturizados/com padrão repetitivo o modelo v1 disparava falso positivo. v2 deve estar melhor — verificar com `slice_metrics.py` em `brightness=escuro`.
4. **Notas muito sobrepostas** (cobertura > 50%) provavelmente serão classificadas como uma só.
5. **Velocidade:** YOLOv8n no Pi4 ≈ 2-4 FPS (medir em hardware alvo).

## Como Reproduzir

```bash
# 1. Setup
git clone <repo>
cd money_pick_n_place
pip install -r requirements.txt
wandb login

# 2. Dataset (se você tem o dvc remote)
dvc pull

# OU (sem DVC) baixar do Roboflow:
python download_dataset.py
python prepare_v2_dataset.py
python freeze_holdout.py

# 3. Treino
python train_v2.py --seed 42

# 4. Avaliação
python evaluate_model.py --model runs/detect/v2_*/weights/best.pt
python slice_metrics.py --model runs/detect/v2_*/weights/best.pt
python find_bad_labels.py --model runs/detect/v2_*/weights/best.pt
```

## Histórico de Versões

| Versão | Data | Dataset | mAP@50 | mAP@50-95 | Notas |
|---|---|---|---|---|---|
| v1 | 2026-04-23 | v1 (cedulas) | 0.964 | 0.864 | Baseline. R$100 fraca, detecta no fundo. |
| v2 | **[PREENCHER]** | v2 (7451 imgs) | **[PREENCHER]** | **[PREENCHER]** | **[PREENCHER]** |

## Contato

- Issues / bugs: **[GitHub issues link]**
- Email: gabriel.dallape@norfolk.ai
