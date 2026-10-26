import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_medical_report(patient_features, prediction_proba, clinical_threshold, is_malignant, shap_top_features):
    """
    Gera um laudo médico estruturado.
    Utiliza a API da OpenAI se configurada (GPT-4o-mini).
    Caso a chave não esteja presente ou ocorra erro de conexão, 
    utiliza um motor de Inteligência Clínica Determinística (Fallback)
    que garante funcionamento e robustez 100% autônoma.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Formatar o bloco de SHAP
    shap_text = "\n".join([f"  • {feat}: impacto de {impact:+.4f} no log-odds" for feat, impact in shap_top_features])
    
    prompt = f"""
Você é um assistente especialista em Oncologia e Inteligência Artificial na Saúde.
Sua missão é gerar um Laudo Técnico de Triagem Hospitalar interpretando os dados matemáticos do modelo de Machine Learning otimizado por Algoritmos Genéticos.

DADOS DA PACIENTE:
- Probabilidade Estimada de Malignidade: {prediction_proba * 100:.2f}%
- Limiar de Decisão Clínico Otimizado (Threshold): {clinical_threshold * 100:.2f}%
- Classificação Final da IA: {'SUSPEITA DE MALIGNIDADE (ALTO RISCO)' if is_malignant else 'PROVÁVEL BENIGNO (BAIXO RISCO)'}
- Principais biomarcadores celulares com maior contribuição SHAP:
{shap_text}
- Biometria enviada: {patient_features}

Instruções para o laudo:
1. Resumo Diagnóstico da Triagem (claramente indicando o risco e comparação com o limiar).
2. Justificativa Matemática e Explicabilidade (análise das features morfológicas apontadas pelo SHAP).
3. Conduta Clínica Sugerida (orientação para o médico: se biópsia com agulha grossa, mamografia diagnóstica ou seguimento anual de rotina).
Use tom estritamente profissional, técnico e com foco na segurança do paciente.
"""

    if api_key and api_key != "cole_sua_chave_aqui" and len(api_key.strip()) > 10:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            logging.info("Consultando OpenAI GPT-4o-mini para explicabilidade clínica...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um assistente oncológico clínico e técnico de alta precisão."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Falha na chamada OpenAI ({e}). Acionando gerador clínico determinístico de segurança.")
    
    # Fallback Determinístico Clínico de Alta Fidelidade (Garante funcionamento sem necessidade de API externa)
    status_label = "ALTO RISCO - INDICAÇÃO DE INVESTIGAÇÃO URGENTE" if is_malignant else "BAIXO RISCO - ACHADO PROVAVELMENTE BENIGNO"
    top_driver = shap_top_features[0][0] if shap_top_features else "morfologia celular"
    
    report = f"""
### 🏥 LAUDO TÉCNICO DE TRIAGEM ONCOLÓGICA (SISTEMA OTIMIZADO POR AG)

**1. Resumo Clínico e Probabilidade Preditiva:**
- **Probabilidade calculada pela IA:** `{prediction_proba * 100:.2f}%`
- **Limiar de Decisão Clínico (Cutoff Genético):** `{clinical_threshold * 100:.2f}%`
- **Classificação:** **{status_label}**
*(O modelo operou sob um limiar otimizado de segurança clínica de modo a maximizar a taxa de recall e minimizar falsos negativos).*

**2. Explicabilidade dos Biomarcadores (Análise SHAP):**
A decisão algorítmica foi orientada pelas seguintes deformidades morfológicas nucleares:
{shap_text}

*Interpretação:* A variável `{top_driver}` foi o fator mais determinante para o posicionamento da probabilidade em relação à fronteira de decisão. Anormalidades em raio, concavidade e perímetro celular são marcadores clássicos de pleomorfismo nuclear associados a lesões proliferativas.

**3. Conduta Médica Recomendada:**
"""
    if is_malignant:
        report += """- **Recomendação:** Encaminhamento para exame complementar urgente (Core Biopsy guiada por ultrassom ou mamografia diagnóstica com magnificação).
- **Prioridade:** **Alta** (Triagem prioritária para confirmação histopatológica)."""
    else:
        report += """- **Recomendação:** Manutenção do rastreamento de rotina periódico conforme diretrizes para a faixa etária.
- **Prioridade:** Regular / Acompanhamento ambulatorial."""

    return report.strip()
