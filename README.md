# 🧬 FIAP Tech Challenge - Fase 2: Ecossistema Integrado de Triagem Oncológica

**Projeto 1 — Otimização de Modelos de Diagnóstico Oncológico via Algoritmos Genéticos e IA Generativa (LLMs)**

> **Integrantes (Grupo 40):** Felipe Antonio Ayancan Hernandez & Fernando Ponz Vargas  
> **Vídeo de Demonstração:** [Inserir Link do YouTube/Vimeo]  
> **Documentação de Arquitetura & Evidências K8s:** [`docs/arquitetura.md`](docs/arquitetura.md)  
> **Relatório Formal de Entrega:** [`Relatorio_Entrega_Fase2.docx`](Relatorio_Entrega_Fase2.docx)

---

## 🎯 Visão Geral & Unificação dos Dois Mundos

Este repositório consolida a integração completa entre **Engenharia de Machine Learning**, **IA Generativa / Explicabilidade (SHAP)**, **DevOps/MLOps em Nuvem com Kubernetes** e uma **Interface Clínica Interativa (Streamlit)** para profissionais da saúde.

```text
                                  ┌───────────────────────────┐
                                  │   Médico / Especialista   │
                                  └─────────────┬─────────────┘
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
          ┌─────────────────────────────┐               ┌─────────────────────────────┐
          │      Frontend Web UI        │               │      Sistemas Externos      │
          │         (Streamlit)         │               │     (Consumo via HTTP)      │
          │           app.py            │               │                             │
          └──────────────┬──────────────┘               └──────────────┬──────────────┘
                         │                                             │
                         │                                             │
                         └──────────────────────┬──────────────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │      Backend REST         │
                                  │        (FastAPI)          │
                                  │    /predict, /interpret   │
                                  └─────────────┬─────────────┘
                                                │
                   ┌────────────────────────────┼────────────────────────────┐
                   ▼                            ▼                            ▼
         ┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
         │     Modelo ML     │        │    Explicabilidade│        │   Interpretação   │
         │   Otimizado por   │        │     com SHAP      │        │      com LLM      │
         │ Algoritmo Genético│        │  (Fatores Locais) │        │ (Laudo Estruturado│
         │    (DEAP / C=0.1) │        │                   │        │  + Fallback Auto) │
         └─────────┬─────────┘        └─────────┬─────────┘        └─────────┬─────────┘
                   │                            │                            │
                   └────────────────────────────┼────────────────────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ Infraestrutura & DevOps   │
                                  │ Docker + K8s HPA (1 a 4)  │
                                  │ Monitoramento /metrics    │
                                  └───────────────────────────┘
```

---

## 📊 Resultados Principais no Teste Cego

| Indicador (Teste: 114 casos) | Fase 1 (GridSearchCV) | Fase 2 (Algoritmo Genético + Threshold) | Impacto Clínico |
| :--- | :---: | :---: | :--- |
| **Acurácia** | 0,9737 | **0,9912** | Redução substancial de erros globais |
| **Precisão** | 0,9756 | **1,0000** | Zero falsos positivos no teste cego |
| **Recall (Maligno)** | 0,9524 | **0,9762** | **Máxima sensibilidade diagnóstica** |
| **F1-Score** | 0,9639 | **0,9880** | Harmonia estatística superior |
| **ROC-AUC** | 0,9954 | **0,9977** | Calibração de probabilidade quase perfeita |
| **Falsos Negativos (Risco)**| 2 pacientes | **1 paciente** | **Redução de 50% a 66% no risco de subdiagnóstico** |

- **Algoritmo Genético (DEAP):** 3 experimentos completos avaliados, convergindo para `C = 0.1`, `class_weight = balanced` e penalidade `l2`.
- **Threshold Clínico:** Limiar ajustado out-of-fold com base no feedback do professor da Fase 1, aumentando o colchão de proteção contra falsos negativos.
- **SHAP:** Extração dos principais biomarcadores determinantes em nível celular.
- **IA Generativa (LLM):** Transforma números em laudos médicos claros para os oncologistas, com fallback inteligente determinístico para garantir operação mesmo sem conexão externa.
- **Escalabilidade Automática (Kubernetes HPA):** Testado sob carga e evidenciado com autoscaling de **1 → 2 → 4 réplicas** de pods.

---

## 📂 Estrutura do Projeto Unificado

```text
FIAP_TechChallenge_02_2026/
├── api/                        # Backend REST (FastAPI) com endpoints /predict, /interpret, /metrics
├── src/                        # Núcleo modular de serviços (Predição, SHAP, LLM, AG)
│   ├── prediction_service.py   # Serviço de predição do modelo
│   ├── explainability_service.py# Serviço SHAP
│   ├── llm_service.py          # Serviço de linguagem natural
│   ├── genetic_optimizer.py    # Algoritmo Genético (DEAP) com 3 experimentos
│   ├── llm_explainer.py        # Motor de laudos clínicos com fallback
│   └── model_trainer.py        # Pipelines de treinamento
├── notebooks/                  # Cadernos de experimentação e validação detalhados
│   ├── 01_baseline_fase1.ipynb
│   ├── 02_algoritmo_genetico.ipynb
│   ├── 03_threshold_e_explicabilidade.ipynb
│   └── 04_llm.ipynb
├── k8s/                        # Manifestos Kubernetes (Deployment, Service, HPA, ConfigMap)
├── models/                     # Modelos treinados (.joblib) e metadados de features
├── docs/                       # Documentação técnica profunda (arquitetura.md) e prints de evidências K8s
├── tests/                      # Bateria de testes automatizados unitários e de integração
├── app.py                      # Frontend Web Interativo em Streamlit para uso clínico
├── tech_challenge_fase2.ipynb  # Notebook didático unificado da entrega
├── Relatorio_Entrega_Fase2.docx# Relatório técnico formal da Fase 2 para a FIAP
├── docker-compose.yml          # Orquestração local de containers
├── Dockerfile                  # Container pronto para produção
├── projeto.bat                 # Script de automação e controle local
└── requirements.txt            # Dependências fixadas e unificadas
```

---

## 🚀 Como Executar

### 1. Iniciar o Frontend Web (Streamlit)
```bash
streamlit run app.py
```
Acesse: `http://localhost:8501` para interagir com o assistente médico e gerar laudos em tempo real.

### 2. Iniciar a API REST (FastAPI)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Documentação interativa disponível em: `http://localhost:8000/docs`.

### 3. Executar os Testes Automatizados
```bash
pytest tests/
```

### 4. Orquestração Local ou Kubernetes
- **Docker Compose:** `docker-compose up --build`
- **Menu Rápido no Windows:** executar `projeto.bat`
- **Kubernetes:** aplicar os manifestos em `k8s/`
