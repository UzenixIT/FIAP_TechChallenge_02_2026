# 🧬 FIAP Tech Challenge - Fase 2: Otimização Genética & Explicabilidade Clínica com LLMs

Repositório da **Fase 2 do Tech Challenge** - Pós-Graduação em Inteligência Artificial e Dados (FIAP).

> **Integrantes (Grupo 40):** Felipe Antonio Ayancan Hernandez e Fernando Ponz Vargas  
> **Tema:** Otimização de Modelos de Diagnóstico Oncológico (Projeto 1)

---

## 🎯 Contextualização & Continuidade da Fase 1

Na **Fase 1**, desenvolvemos modelos para triagem oncológica de câncer de mama (*Breast Cancer Wisconsin Diagnostic*), obtendo **100% de aproveitamento** com menção honrosa ao rigor analítico e explicabilidade via SHAP.

O professor orientou o seguinte próximo passo em seu parecer:
> *"Experimente extrair as probabilidades (`predict_proba`) da sua Regressão Logística otimizada e reduzir o limiar de decisão de 0,5 para 0,3 para verificar se os falsos negativos restantes no teste zeram, aumentando o colchão de segurança da triagem hospitalar."*

Na **Fase 2**, automatizamos e expandimos essa recomendação:
1. **Algoritmos Genéticos (DEAP):** O limiar de decisão clínico (*Threshold*) e os hiperparâmetros de regularização ($C$, penalidade, balanceamento de classes) foram modelados em um **cromossomo conjunto** submetido a seleção, cruzamento e mutação.
2. **Fitness Function Clínica:** Foco rigoroso em maximizar o **Recall** ($80\%$) com desempate em **F1-Score** ($20\%$), minimizando ao máximo os falsos negativos malignos.
3. **Explicabilidade com IA Generativa (LLM):** O modelo transmite a probabilidade calibrada e os biomarcadores com maior impacto no SHAP para que um agente LLM gere **laudos oncológicos em linguagem natural** para a equipe médica.
4. **Aplicação Web Interativa (Streamlit):** Painel clínico para simulação em tempo real de pacientes com geração instantânea do laudo.

---

## 📊 Resultados dos Experimentos Genéticos

Realizamos **3 experimentos completos** variando tamanho de população, taxas de cruzamento e mutação:

| Experimento | População | Gerações | Crossover | Mutação | Fitness CV | Melhor $C$ | Class Weight | Threshold Clínico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1 (Conservador)** | 20 | 10 | 50% | 20% | 0.9757 | 4.3179 | None | 0.1677 |
| **Exp 2 (Alta Exploração)** 🏆 | **35** | **15** | **70%** | **40%** | **0.9804** | **0.9307** | **None** | **0.1815** |
| **Exp 3 (População Expandida)** | 50 | 20 | 60% | 25% | 0.9767 | 1.9221 | balanced | 0.2932 |

### 📈 Comparativo de Desempenho no Teste Cego: Fase 1 vs. Fase 2

| Métrica | Fase 1 (Original, $T=0.50$) | Fase 2 (AG + Threshold Clínico $T=0.18$) | Impacto Clínico |
| :--- | :---: | :---: | :--- |
| **Recall (Maligno)** | **92.86%** | **97.62%** | 🚀 **+4.76% de sensibilidade diagnóstica** |
| **Falsos Negativos (FN)** | **3 casos** | **1 caso** | 📉 **Redução de 66.7% no risco de subdiagnóstico** |
| **Precisão** | 97.50% | 91.11% | Margem de segurança controlada |
| **F1-Score** | 95.12% | 94.25% | Equilíbrio mantido em nível de excelência |
| **Acurácia Geral** | 96.49% | 95.61% | Elevada confiabilidade diagnóstica |

---

## 🏗️ Estrutura do Repositório

```text
FIAP_TechChallenge_02_2026/
├── app.py                      # Aplicação web Streamlit com inferência real + SHAP + LLM
├── requirements.txt            # Dependências fixadas e compatíveis
├── Dockerfile                  # Containerização pronta para produção
├── railway.json                # Deploy em nuvem (Railway)
├── tech_challenge_fase2.ipynb  # Notebook estruturado com os 3 experimentos e gráficos
├── generate_nb.py              # Script utilitário para geração limpa do notebook
├── data/
│   ├── data.csv                # Dataset Breast Cancer Wisconsin
│   ├── best_model_phase2.joblib# Pipeline treinado e calibrado pelo melhor indivíduo AG
│   └── experiments_results.joblib # Histórico completo de convergência dos experimentos
└── src/
    ├── model_trainer.py        # Pipelines de ML e pré-processamento
    ├── genetic_optimizer.py    # Algoritmo Genético (DEAP) com 3 experimentos e fitness clínica
    └── llm_explainer.py        # Motor de explicabilidade com LLM (OpenAI + Fallback Determinístico)
```

---

## 🚀 Como Executar o Projeto

### 1. Criar e Ativar o Ambiente Virtual

```bash
python -m venv .venv
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Rodar os Experimentos do Algoritmo Genético

```bash
python -m src.genetic_optimizer
```

### 4. Iniciar a Aplicação Interativa (Streamlit)

```bash
streamlit run app.py
```

Acesse no navegador: `http://localhost:8501`.

---

## 🛡️ Explicabilidade com LLMs & Fallback de Segurança

O módulo `src/llm_explainer.py` suporta:
1. **OpenAI GPT-4o-mini:** Ativado automaticamente se a variável `OPENAI_API_KEY` estiver no `.env`.
2. **Motor Clínico Determinístico:** Caso a API externa esteja indisponível ou sem chave configurada, um gerador oncológico estruturado assume a geração do laudo instantaneamente, garantindo **resiliência total sem falhas na apresentação**.
