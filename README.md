# Proposta — Detecção de Ataques com Federated Learning + SDN

**Projeto:** Iniciação Científica  
**Data:** Junho de 2026  
**Status:** Prova de conceito FL concluída — integração SDN em desenvolvimento  

---

## 1. Visão Geral

Proposta de arquitetura que combina **Federated Learning (FL)** e **Software-Defined Networking (SDN)** para detecção distribuída de ataques em redes, com resposta automática de mitigação via controlador SDN.

O diferencial em relação a abordagens centralizadas:

- Dados de tráfego **nunca saem dos nós** — apenas pesos do modelo trafegam
- Detecção **distribuída e colaborativa** entre múltiplos pontos da rede
- **Resposta automática** via controlador SDN ao detectar ataque
- Resistência a **dois vetores de ataque simultâneos**: na rede e no modelo FL

---

## 2. Cenários de Ataque Cobertos

| Vetor | Tipo | Detecção |
|---|---|---|
| Rede SDN | DDoS / flooding / port scan | Modelo FL classifica métricas de tráfego |
| Modelo FL | Envenenamento (model poisoning) | Estratégia de agregação robusta (FedProx / Krum) |

---

## 3. Arquitetura

```
        ┌─────────────────────────────────────────┐
        │             Camada SDN                  │
        │                                         │
        │  [Switch OF-1]   [Controlador RYU]  [Switch OF-2]  │
        │       │          API REST /stats         │     │
        └───────┼──────────────┬──────────────────┼─────┘
                │              │ alertas           │
        ┌───────┴──────┐       │       ┌───────────┴──────┐
        │  Cliente FL 1 │       │       │  Cliente FL 2    │
        │  Treino local │       │       │  Treino local    │
        │  Detector IDS │       │       │  Detector IDS    │
        │  (normal/atq) │       │       │  (normal/atq)    │
        └───────┬───────┘       │       └────────┬─────────┘
                │   pesos       │         pesos  │
                └───────────────┼────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │      Servidor FL       │
                    │  Agregador (FedAvg)   │
                    │  Módulo de Mitigação  │
                    │  → notifica RYU       │
                    └───────────────────────┘
```

### Fluxo por rodada

1. Controlador RYU coleta métricas dos switches (bytes/s, pacotes/s, nº fluxos)
2. Cada cliente FL recebe as métricas do seu switch local
3. Clientes treinam o classificador localmente (normal vs. ataque)
4. Pesos são enviados ao servidor FL
5. Servidor agrega via FedAvg e gera modelo global
6. Se detecção de ataque → servidor notifica RYU via REST
7. RYU instala regra de bloqueio no switch (`/stats/flowentry/add`)

---

## 4. Stack Tecnológica

| Componente | Ferramenta |
|---|---|
| Framework FL | Flower (flwr) |
| Deep Learning | PyTorch |
| Rede SDN | Mininet + switches OpenFlow |
| Controlador SDN | RYU Controller |
| Dataset | CIC-IDS2017 ou NSL-KDD |
| Comunicação SDN↔FL | REST API (requests) |
| Linguagem | Python 3.10 |

### Instalação completa

```bash
pip install flwr torch torchvision pandas scikit-learn requests
sudo apt install mininet
pip install ryu
```

---

## 5. Modelo de Detecção

### Mudanças em relação à PoC atual

| Aspecto | PoC atual | Nova proposta |
|---|---|---|
| Dataset | MNIST (imagens) | CIC-IDS2017 (tráfego de rede) |
| Tarefa | Classificação 10 classes | Classificação binária (normal / ataque) |
| Features | Pixels 28×28 | Métricas de fluxo SDN |
| Saída do cliente | Pesos CNN | Pesos CNN + alerta ao servidor |

### Features de entrada (métricas SDN)

- Bytes por segundo por fluxo
- Pacotes por segundo por fluxo
- Número de fluxos ativos no switch
- Duração média dos fluxos
- Razão TCP SYN / SYN-ACK (indicador de DDoS)
- Entropia dos IPs de destino (indicador de port scan)

### Arquitetura do modelo

```
Input (N features) → FC(128) → ReLU → Dropout(0.3)
                  → FC(64)  → ReLU
                  → FC(2)   → Softmax
                  → [normal | ataque]
```

---

## 6. Dataset

### CIC-IDS2017 (recomendado)

- Gerado pela Universidade do New Brunswick
- Contém tráfego normal + DDoS, PortScan, Brute Force, Web Attack
- ~2,8 milhões de amostras rotuladas
- Amplamente citado em literatura de IDS

### Particionamento para FL

Cada cliente recebe uma fatia do dataset simulando **dados heterogêneos (Non-IID)**:

- Cliente 1: tráfego do segmento de rede 1 (mais tráfego normal)
- Cliente 2: tráfego do segmento de rede 2 (mais tráfego de ataque)

Isso simula a realidade de sensores em pontos distintos de uma rede corporativa ou IoT.

---

## 7. Módulo de Mitigação

Quando o modelo global (após agregação FedAvg) classifica um padrão como ataque, o servidor FL envia um alerta ao controlador RYU via REST:

```python
# Exemplo: bloquear IP malicioso via RYU REST API
import requests

def mitigar_ataque(ip_origem, switch_dpid):
    flow = {
        "dpid": switch_dpid,
        "priority": 100,
        "match": {"nw_src": ip_origem, "dl_type": 2048},
        "actions": []  # lista vazia = DROP
    }
    requests.post("http://localhost:8080/stats/flowentry/add",
                  json=flow)
```

O switch instala a regra e descarta pacotes do IP malicioso até a próxima rodada de avaliação.

---

## 8. Resistência a Model Poisoning

Para cobrir o segundo vetor de ataque (cliente malicioso enviando pesos envenenados), a estratégia de agregação pode ser substituída ou complementada:

| Estratégia | Proteção |
|---|---|
| FedAvg (baseline) | Nenhuma proteção contra poisoning |
| FedProx | Penaliza clientes muito divergentes |
| Krum / Multi-Krum | Descarta contribuições anômalas |
| Detecção de anomalia nos pesos | Score de similaridade entre clientes |

Para a IC, o caminho recomendado é implementar FedAvg como baseline e FedProx como comparação, com um experimento de poisoning simulado (um cliente enviando pesos aleatórios).

---

## 9. Experimentos Planejados

| Experimento | Objetivo |
|---|---|
| E1 — Baseline | FL + CIC-IDS2017, sem SDN, FedAvg, dados IID |
| E2 — Non-IID | Mesmo modelo com particionamento heterogêneo |
| E3 — Integração SDN | FL + Mininet/RYU, métricas reais de fluxo |
| E4 — Ataque na rede | Simular DDoS no Mininet, verificar detecção |
| E5 — Model poisoning | 1 cliente malicioso, comparar FedAvg vs FedProx |
| E6 — Mitigação | Loop completo: detecção → alerta → bloqueio SDN |

---

## 10. Próximos Passos Imediatos

1. Baixar e pré-processar o CIC-IDS2017
2. Adaptar `model.py` para classificador binário de tráfego
3. Adaptar `client.py` para carregar features de rede
4. Instalar e configurar Mininet + RYU
5. Implementar coleta de métricas via REST API do RYU
6. Implementar módulo de mitigação no `server.py`
7. Rodar Experimento E1 como novo baseline

---

## 11. Situação Atual da PoC

A prova de conceito de FL com Flower + PyTorch + MNIST foi concluída com sucesso:

| Rodada | Loss | Acurácia |
|--------|------|----------|
| 1 | 0.0848 | 98,50% |
| 2 | 0.0361 | 98,97% |
| 3 | 0.0221 | 99,26% |

- 2 clientes, 3 rodadas, FedAvg, sem falhas
- Dados nunca saíram dos clientes
- Base de código pronta para extensão com SDN e novo dataset

---

*Proposta elaborada como evolução da PoC inicial de Federated Learning para Iniciação Científica em segurança de redes com FL + SDN.*
