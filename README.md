# SQLi Security Scan

Ferramenta de detecção de vulnerabilidades de SQL Injection em APIs REST utilizando payloads gerados por uma GAN baseada em LSTM.

A ferramenta analisa a especificação Swagger/OpenAPI da API alvo, gera payloads de SQL Injection contextualizados para cada endpoint e parâmetro, executa os ataques e classifica os resultados automaticamente.

## Requisitos

- Python 3.10+
- CUDA (opcional, recomendado para treinamento)

## Instalação

```bash
pip install -r requirements.txt
```

Dependências principais:
- `torch` >= 2.0.0
- `numpy` >= 1.24.0
- `pandas` >= 2.0.0
- `requests` >= 2.31.0
- `pyyaml` >= 6.0
- `colorama` >= 0.4.6

## Uso

A ferramenta possui dois comandos: `train` e `scan`.

```bash
python main.py <comando> [opções]
```

---

### `train` — Treinar o modelo GAN

Treina a GAN nos datasets de SQL Injection disponíveis no diretório `datasets/`.

```bash
python main.py train [opções]
```

#### Parâmetros

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--datasets-dir` | `datasets` | Diretório contendo os datasets CSV/TXT |
| `--output-dir` | `models` | Diretório para salvar o modelo treinado |
| `--epochs` | `500` | Número de épocas de treinamento |
| `--batch-size` | `256` | Tamanho do batch |
| `--resume` | — | Caminho de um checkpoint para continuar o treinamento |

#### Exemplos

```bash
# Treinar com configuração padrão
python main.py train

# Treinar com parâmetros customizados
python main.py train --epochs 300 --batch-size 128

# Continuar treinamento a partir de um checkpoint
python main.py train --resume models/checkpoint_epoch_200.pt
```

#### Configuração interna do modelo

| Parâmetro | Valor |
|-----------|-------|
| `embed_dim` | 128 |
| `hidden_dim` | 512 |
| `num_layers` | 3 |
| `max_len` | 256 |
| `lr_gen` | 1e-4 |
| `lr_disc` | 3e-4 |
| `teacher_forcing_ratio` | 0.5 |

Checkpoints são salvos automaticamente a cada 50 épocas em `<output-dir>/checkpoint_epoch_<N>.pt`. O modelo final é salvo como `<output-dir>/gan_sqli.pt`.

---

### `scan` — Executar scan de vulnerabilidades

Escaneia uma API REST usando payloads gerados pela GAN treinada.

```bash
python main.py scan --swagger <caminho> [opções]
```

#### Parâmetros

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--swagger` | **(obrigatório)** | Caminho do arquivo Swagger/OpenAPI (YAML ou JSON) |
| `--base-url` | extraído do Swagger | URL base da API (sobrescreve o valor do Swagger) |
| `--model-path` | `models/gan_sqli.pt` | Caminho do modelo GAN treinado |
| `--num-payloads` | `200` | Número de payloads a gerar por endpoint |
| `--temperature` | `0.7` | Temperatura da geração (maior = mais variação) |
| `--output` | `reports/scan_report.json` | Caminho do relatório JSON de saída |
| `--auth-token` | — | Token de autenticação (ex: `Token abc123` ou `Bearer xyz`) |

#### Exemplos

```bash
# Scan básico
python main.py scan --swagger docs/api-swagger.yaml

# Scan com autenticação e URL customizada
python main.py scan \
  --swagger docs/api-swagger.yaml \
  --base-url http://localhost:8000/api \
  --auth-token "Token meu_token_aqui"

# Scan com mais payloads e maior variação
python main.py scan \
  --swagger docs/api-swagger.yaml \
  --num-payloads 500 \
  --temperature 0.9

# Scan usando modelo específico
python main.py scan \
  --swagger docs/api-swagger.yaml \
  --model-path models/checkpoint_epoch_300.pt
```

---

## Pipeline de Execução

```
1. Swagger/OpenAPI  →  Parser extrai endpoints e parâmetros
2. GAN treinada     →  Gera payloads SQLi contextualizados por tipo de parâmetro
3. Attacker         →  Injeta payloads via path, query, body e headers
4. Analyzer         →  Classifica respostas (VP, FP, FALHA, ERRO)
5. Report           →  Gera relatório JSON com métricas
```

### Classificação de resultados

| Sigla | Significado | Descrição |
|-------|-------------|-----------|
| VP | Verdadeiro Positivo | Vulnerabilidade de SQLi detectada (erros de DB, vazamento de dados, stack traces) |
| FP | Falso Positivo | Resposta suspeita, mas sem indicadores concretos |
| FALHA | Falha | O servidor rejeitou o payload (sem vulnerabilidade) |
| ERRO | Erro | Erro de conexão ou timeout |

### Métricas do relatório

- **precision** — VP / (VP + FP)
- **efficacy** — VP / total de ataques
- **total_attacks** — total de payloads enviados
- **true_positives** / **false_positives** / **failures** / **errors**
- **execution_time_seconds**

---

## Datasets

O diretório `datasets/` deve conter arquivos CSV com payloads de SQL Injection. Formatos suportados:

| Arquivo | Formato |
|---------|---------|
| `*_payload_full.csv` | Colunas: `payload`, `attack_type` (`sqli` / `norm`) |
| `SQLI_Dataset.csv` | Colunas: `Query`, `Label` (`1` = malicioso) |
| `sqli-extended.csv` | Colunas: `Query`, `Label` |
| `*.txt` | Um payload por linha (todos tratados como maliciosos) |

---

## Estrutura do Projeto

```
SQLi Security Scan/
├── main.py                  # CLI principal (train / scan)
├── requirements.txt         # Dependências Python
├── datasets/                # Datasets de SQL Injection
├── models/                  # Modelos treinados e checkpoints
├── reports/                 # Relatórios de scan gerados
└── scanner/
    ├── __init__.py
    ├── analyzer.py          # Classificação de respostas HTTP
    ├── attacker.py          # Execução dos ataques nos endpoints
    ├── payloads.py          # Carregamento e geração de payloads
    ├── report.py            # Geração de relatórios JSON
    ├── swagger_parser.py    # Parser de Swagger/OpenAPI
    └── gan/
        ├── __init__.py
        ├── models.py        # Arquiteturas Generator e Discriminator (LSTM)
        ├── preprocessing.py # Limpeza e encoding dos datasets
        ├── train.py         # Loop de treinamento da GAN
        └── generate.py      # Geração de payloads com o Generator treinado
```

---
