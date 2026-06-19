# Prova de Conceito — Federated Learning com Flower e PyTorch

**Data:** Junho de 2026  
**Framework:** Flower (flwr) + PyTorch + torchvision  
**Dataset:** MNIST  

---

## 1. Objetivo

Implementar e validar uma prova de conceito de **Aprendizado Federado (Federated Learning — FL)** usando o framework Flower, com dois clientes treinando localmente sobre o dataset MNIST e um servidor central agregando os modelos via algoritmo **FedAvg**.

---

## 2. Ambiente e Dependências

| Componente | Versão / Detalhe |
|---|---|
| Sistema operacional | Ubuntu (Linux) |
| Python | 3.10 |
| Framework FL | Flower (`flwr`) |
| Framework DL | PyTorch + torchvision |
| Dataset | MNIST (60.000 treino / 10.000 teste) |
| Hardware | CPU (sem GPU) |

### Instalação

```bash
python3 -m venv fl_env
source fl_env/bin/activate
pip install flwr torch torchvision
```

---

## 3. Arquitetura do Experimento

```
        Servidor Flower (FedAvg)
               │
    ┌──────────┴──────────┐
    │                     │
 Cliente 1             Cliente 2
 (MNIST local)         (MNIST local)
 Treino local          Treino local
```

**Fluxo por rodada:**

1. Servidor envia pesos globais para todos os clientes
2. Cada cliente treina localmente por 1 época
3. Clientes retornam pesos atualizados ao servidor
4. Servidor aplica FedAvg e gera novo modelo global
5. Processo repete por `num_rounds` rodadas

---

## 4. Implementação

### 4.1 Modelo (`model.py`)

CNN com duas camadas convolucionais para classificação de dígitos MNIST:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1   = nn.Linear(64 * 7 * 7, 512)
        self.fc2   = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

**Arquitetura:** Conv(1→32) → MaxPool → Conv(32→64) → MaxPool → FC(3136→512) → FC(512→10)

### 4.2 Cliente (`client.py`)

Cada cliente implementa `fl.client.NumPyClient` com três métodos:

- `get_parameters` — serializa os pesos do modelo para envio ao servidor
- `fit` — recebe pesos globais, treina localmente (Adam, lr=0.001, batch=32), retorna pesos atualizados
- `evaluate` — avalia o modelo global no conjunto de teste local e retorna loss e accuracy

### 4.3 Servidor (`server.py`)

```python
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
```

**Agregação FedAvg:**

$$w_G = \frac{1}{\sum_{k \in K}|D_k|} \sum_{k=1}^{K} |D_k| \cdot w_k$$

onde $w_k$ são os pesos do cliente $k$ e $|D_k|$ é o tamanho do seu dataset local.

---

## 5. Resultados

### Experimento 1 (sem agregação de métricas)

3 rodadas, 2 clientes, sem `evaluate_metrics_aggregation_fn`:

| Rodada | Loss distribuído |
|--------|-----------------|
| 1 | 0.06147 |
| 2 | 0.02764 |
| 3 | 0.02258 |

Tempo total: **326,54 segundos**

### Experimento 2 (com agregação de métricas)

3 rodadas, 2 clientes, com `evaluate_metrics_aggregation_fn=weighted_average`:

| Rodada | Loss distribuído | Acurácia global |
|--------|-----------------|-----------------|
| 1 | 0.08481 | 98,50% |
| 2 | 0.03608 | 98,97% |
| 3 | 0.02206 | **99,26%** |

Tempo total: **311,21 segundos**

### Evolução da acurácia

```
Rodada 1 ████████████████████████████████████████████████░ 98,50%
Rodada 2 ████████████████████████████████████████████████░ 98,97%
Rodada 3 █████████████████████████████████████████████████ 99,26%
```

---

## 6. Análise dos Resultados

- O loss caiu **~63%** entre a rodada 1 e a rodada 3, demonstrando convergência estável do modelo global.
- A acurácia atingiu **99,26%** na rodada 3, resultado competitivo com treino centralizado no MNIST.
- Em nenhuma rodada houve falha de cliente (`0 failures`), confirmando estabilidade da comunicação local via gRPC.
- Os dados **nunca saíram dos clientes** — apenas os pesos do modelo trafegaram pela rede, validando a premissa de privacidade do FL.

---

## 7. Observações Técnicas

- A versão do Flower utilizada apresenta warnings de APIs depreciadas (`start_server`, `start_numpy_client`). A funcionalidade não é afetada, mas a migração para a API nova (`flower-superlink` / `flower-supernode`) será necessária em versões futuras.
- O download do MNIST pelo torchvision falhou nos servidores oficiais (HTTP 404). Solução: download manual via `storage.googleapis.com/cvdf-datasets/mnist/`.
- O treinamento foi executado inteiramente em CPU, justificando o tempo de ~5 minutos por experimento.

---

## 8. Próximos Passos

| Etapa | Descrição |
|---|---|
| Non-IID data | Particionar MNIST de forma heterogênea para simular dados reais de IoT |
| Mais clientes | Usar `flwr.simulation` para simular 10, 50 ou 100 clientes |
| FedProx | Comparar convergência com FedAvg em cenários Non-IID |
| Differential Privacy | Adicionar DP-SGD para privacidade formal |
| Métricas por rodada | Gerar gráficos de loss e accuracy para publicação |
| Edge devices | Migrar clientes para Raspberry Pi ou dispositivos IoT reais |

---

## 9. Estrutura do Projeto

```
IC2 - FL/
├── model.py      # CNN (2 conv + 2 FC)
├── client.py     # Cliente Flower com treino local
├── server.py     # Servidor Flower com FedAvg
├── fl_env/       # Ambiente virtual Python
└── MNIST/        # Dataset baixado localmente
    └── raw/
```

---

*Experimento realizado como prova de conceito para Iniciação Científica em Federated Learning aplicado a IoT.*
