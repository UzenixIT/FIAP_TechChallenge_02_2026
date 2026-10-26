import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
from src.llm_explainer import generate_medical_report

st.set_page_config(
    page_title="Triagem Oncológica IA - Fase 2", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregamento do modelo otimizado pelo Algoritmo Genético
@st.cache_resource
def load_optimized_artifacts():
    model_path = os.path.join(os.path.dirname(__file__), "data", "best_model_phase2.joblib")
    if os.path.exists(model_path):
        artifacts = joblib.load(model_path)
        return artifacts
    return None

artifacts = load_optimized_artifacts()

st.title("🧬 Assistente Médico de Triagem Oncológica")
st.markdown("""
**FIAP Tech Challenge - Fase 2**  
Sistema preditivo otimizado por **Algoritmos Genéticos (DEAP)** com threshold de segurança clínica e explicabilidade em linguagem natural por **LLMs e SHAP**.
""")

if artifacts is None:
    st.error("Artefatos do modelo otimizado não encontrados em `data/best_model_phase2.joblib`. Execute o Algoritmo Genético primeiro.")
    st.stop()

pipeline = artifacts["pipeline"]
threshold = artifacts["best_threshold"]
c_param = artifacts["best_c"]
cw_param = artifacts["best_cw"]
feature_names = artifacts["feature_names"]

# Base com médias dos dados reais para compor o vetor completo de 30 variáveis
@st.cache_data
def get_reference_features():
    df_path = os.path.join(os.path.dirname(__file__), "data", "data.csv")
    df = pd.read_csv(df_path).drop(columns=["id", "Unnamed: 32"], errors="ignore")
    return df.drop(columns=["diagnosis"]).mean().to_dict()

ref_means = get_reference_features()

# Barra lateral: Hiperparâmetros do AG & Entrada de dados do paciente
st.sidebar.header("⚙️ Modelo Otimizado por AG")
st.sidebar.info(f"""
- **Algoritmo:** Regressão Logística
- **Parâmetro C:** `{c_param:.4f}`
- **Class Weight:** `{cw_param}`
- **Limiar Clínico Otimizado:** `{threshold * 100:.2f}%`
""")

st.sidebar.header("🔬 Biometria Celular da Paciente")
st.sidebar.markdown("*Ajuste os biomarcadores morfológicos de maior relevância clínica:*")

radius_mean = st.sidebar.slider("Raio Médio (radius_mean)", 6.0, 30.0, float(ref_means.get("radius_mean", 14.12)), 0.1)
texture_mean = st.sidebar.slider("Textura Média (texture_mean)", 9.0, 40.0, float(ref_means.get("texture_mean", 19.28)), 0.1)
perimeter_mean = st.sidebar.slider("Perímetro Médio (perimeter_mean)", 40.0, 190.0, float(ref_means.get("perimeter_mean", 91.96)), 0.5)
area_mean = st.sidebar.slider("Área Média (area_mean)", 140.0, 2500.0, float(ref_means.get("area_mean", 654.88)), 10.0)
concavity_mean = st.sidebar.slider("Concavidade Média (concavity_mean)", 0.0, 0.5, float(ref_means.get("concavity_mean", 0.088)), 0.005)
concave_points_mean = st.sidebar.slider("Pontos Côncavos (concave points_mean)", 0.0, 0.25, float(ref_means.get("concave points_mean", 0.048)), 0.005)

# Interface Principal
col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
col_m1.metric("Raio Médio", f"{radius_mean:.2f}")
col_m2.metric("Textura Média", f"{texture_mean:.2f}")
col_m3.metric("Perímetro", f"{perimeter_mean:.1f}")
col_m4.metric("Área Média", f"{area_mean:.1f}")
col_m5.metric("Concavidade", f"{concavity_mean:.3f}")
col_m6.metric("Pts Côncavos", f"{concave_points_mean:.3f}")

st.markdown("---")

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_button = st.button("🚀 Executar Triagem IA", type="primary", use_container_width=True)

with col_info:
    st.caption("O sistema executará a inferência através do pipeline otimizado, computará as contribuições SHAP locais e gerará o laudo via LLM.")

if run_button:
    with st.spinner("Computando inferência clínica, extraindo SHAP e gerando laudo..."):
        # Montar o vetor completo de entrada
        sample_dict = dict(ref_means)
        sample_dict.update({
            "radius_mean": radius_mean,
            "texture_mean": texture_mean,
            "perimeter_mean": perimeter_mean,
            "area_mean": area_mean,
            "concavity_mean": concavity_mean,
            "concave points_mean": concave_points_mean
        })
        
        sample_df = pd.DataFrame([sample_dict])[feature_names]
        
        # Predição de probabilidade
        prob_malignant = float(pipeline.predict_proba(sample_df)[0, 1])
        is_malignant = prob_malignant >= threshold
        
        # Explicabilidade com SHAP no modelo linear
        scaler = pipeline.named_steps["scaler"]
        model = pipeline.named_steps["model"]
        sample_scaled = scaler.transform(sample_df)
        
        # Contribuição local linear = valor normalizado * coeficiente
        linear_contributions = sample_scaled[0] * model.coef_[0]
        feature_importance = list(zip(feature_names, linear_contributions))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        top_shap = feature_importance[:4]
        
        # Exibição do Resultado
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.subheader("🎯 Resultado da Predição")
            if is_malignant:
                st.error(f"""
                ### ⚠️ RISCO ELEVADO DE MALIGNIDADE
                - **Probabilidade de Malignidade:** `{prob_malignant * 100:.2f}%`
                - **Limiar Clínico de Corte:** `{threshold * 100:.2f}%`
                - **Classificação:** Positivo para triagem prioritária
                """)
            else:
                st.success(f"""
                ### ✅ RISCO BAIXO (PROVÁVEL BENIGNO)
                - **Probabilidade de Malignidade:** `{prob_malignant * 100:.2f}%`
                - **Limiar Clínico de Corte:** `{threshold * 100:.2f}%`
                - **Classificação:** Negativo para triagem prioritária
                """)
                
            st.markdown("#### 🔍 Biomarcadores de Maior Influência (SHAP)")
            shap_df = pd.DataFrame(top_shap, columns=["Biomarcador", "Impacto no Score"])
            st.dataframe(shap_df, use_container_width=True)
            
        with col_res2:
            st.subheader("📋 Laudo Técnico Emitido por IA (LLM)")
            patient_summary = {
                "radius_mean": radius_mean,
                "perimeter_mean": perimeter_mean,
                "concavity_mean": concavity_mean,
                "area_mean": area_mean
            }
            report = generate_medical_report(
                patient_features=patient_summary,
                prediction_proba=prob_malignant,
                clinical_threshold=threshold,
                is_malignant=is_malignant,
                shap_top_features=top_shap
            )
            st.markdown(report)
