import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def create_report():
    doc = docx.Document()
    
    # Configurações de margem
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        
    # Título Principal
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("FIAP - PÓS-GRADUAÇÃO EM IA E DADOS\nTECH CHALLENGE - FASE 2")
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0, 51, 102)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Projeto 1: Otimização de Modelos de Diagnóstico Oncológico via Algoritmos Genéticos e Explicabilidade Clínica com LLMs")
    run_sub.font.name = 'Calibri'
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    
    p_aut = doc.add_paragraph()
    p_aut.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_aut = p_aut.add_run("\nAutores: Felipe Antonio Ayancan Hernandez e Fernando Ponz Vargas (Grupo 40)\nData: Outubro de 2026")
    run_aut.font.name = 'Calibri'
    run_aut.font.size = Pt(11)
    run_aut.font.bold = True
    
    doc.add_paragraph("―" * 55)
    
    # 1. Introdução
    h1 = doc.add_heading("1. Introdução e Conexão com o Feedback da Fase 1", level=1)
    h1.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    
    doc.add_paragraph(
        "Na Fase 1, o grupo desenvolveu uma solução de triagem oncológica baseada em dados clínicos estruturados "
        "(Breast Cancer Wisconsin Diagnostic), alcançando 100% de aproveitamento e destaque pelo rigor estatístico e "
        "avaliação consciente de explicabilidade com SHAP."
    )
    
    doc.add_paragraph(
        "Como recomendação do professor avaliador para os próximos passos de engenharia e segurança médica, foi destacado:\n"
        "«Experimente extrair as probabilidades (predict_proba) da sua Regressão Logística otimizada e reduzir o limiar de "
        "decisão de 0,5 para 0,3 para verificar se os falsos negativos restantes no teste zeram, aumentando o colchão de segurança da triagem hospitalar.»"
    )
    
    doc.add_paragraph(
        "Na presente Fase 2, essa recomendação foi adotada e formalizada matematicamente: o limiar de decisão clínico (Threshold) "
        "deixou de ser ajustado manualmente e passou a ser tratado como um gene contínuo otimizado conjuntamente com os hiperparâmetros "
        "de regularização através de Algoritmos Genéticos (DEAP)."
    )
    
    # 2. Metodologia do Algoritmo Genético
    h2 = doc.add_heading("2. Modelagem e Otimização via Algoritmos Genéticos", level=1)
    h2.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    
    doc.add_paragraph(
        "Para a otimização evolutiva, utilizou-se a biblioteca DEAP com a seguinte arquitetura genética:"
    )
    
    doc.add_paragraph(
        "• Representação Cromossômica: Cada indivíduo é composto pelo trio [C, class_weight, threshold_clinico], "
        "onde C varia continuamente entre 0.05 e 15.0, class_weight é binário (0 = None, 1 = 'balanced') e "
        "o limiar clínico é restrito ao intervalo seguro entre 0.10 e 0.60.\n"
        "• Operadores Genéticos: Seleção por Torneio (tamanho = 3), Cruzamento de Dois Pontos (cxTwoPoint) "
        "e Mutação Gaussiana com perturbação nos genes contínuos.\n"
        "• Função Fitness Clínica: O foco absoluto em oncologia é erradicar o subdiagnóstico (Falsos Negativos). "
        "Assim, definiu-se a fitness composta por 80% Recall e 20% F1-Score em Validação Cruzada Estratificada (5-folds):"
    )
    
    doc.add_paragraph("Fitness = 0.80 × Recall + 0.20 × F1-Score", style='Quote')
    
    # 3. Experimentos
    h3 = doc.add_heading("3. Resultados dos 3 Experimentos Obrigatórios", level=1)
    h3.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    
    doc.add_paragraph(
        "Conforme preconizado nos requisitos do Tech Challenge, foram executados 3 experimentos distintos variando o tamanho da população, "
        "número de gerações e taxas de recombinação genética:"
    )
    
    # Tabela de Experimentos
    table_exp = doc.add_table(rows=4, cols=7)
    table_exp.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_exp.style = 'Light Shading Accent 1'
    
    headers = ["Experimento", "Pop.", "Ger.", "Crossover", "Mutação", "Fitness CV", "Threshold"]
    for i, h in enumerate(headers):
        table_exp.rows[0].cells[i].text = h
        table_exp.rows[0].cells[i].paragraphs[0].runs[0].font.bold = True
        
    data_exp = [
        ["1. Conservador", "20", "10", "50%", "20%", "0.9757", "0.1677"],
        ["2. Alta Exploração (🏆)", "35", "15", "70%", "40%", "0.9804", "0.1815"],
        ["3. População Expandida", "50", "20", "60%", "25%", "0.9767", "0.2932"]
    ]
    
    for row_idx, row_data in enumerate(data_exp):
        for col_idx, text in enumerate(row_data):
            table_exp.rows[row_idx + 1].cells[col_idx].text = text
            
    doc.add_paragraph(
        "\nO Experimento 2 apresentou a maior fitness composta média ao manter maior diversidade genética na população, "
        "convergindo para C = 0.9307 e Limiar Clínico de 18.15%."
    )
    
    # 4. Comparativo Fase 1 vs Fase 2
    h4 = doc.add_heading("4. Comparativo de Desempenho no Teste Cego: Fase 1 vs. Fase 2", level=1)
    h4.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    
    table_comp = doc.add_table(rows=6, cols=3)
    table_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_comp.style = 'Light Grid Accent 1'
    
    headers_c = ["Métrica Avaliada", "Fase 1 (Original, T=0.50)", "Fase 2 (AG + Threshold T=0.18)"]
    for i, h in enumerate(headers_c):
        table_comp.rows[0].cells[i].text = h
        table_comp.rows[0].cells[i].paragraphs[0].runs[0].font.bold = True
        
    data_comp = [
        ["Recall (Maligno)", "92.86%", "97.62% (+4.76%)"],
        ["Falsos Negativos (Risco)", "3 pacientes", "1 paciente (-66.7%)"],
        ["Precisão", "97.50%", "91.11%"],
        ["F1-Score", "95.12%", "94.25%"],
        ["Acurácia Global", "96.49%", "95.61%"]
    ]
    for row_idx, row_data in enumerate(data_comp):
        for col_idx, text in enumerate(row_data):
            table_comp.rows[row_idx + 1].cells[col_idx].text = text
            
    doc.add_paragraph(
        "\nA redução dos falsos negativos de 3 para 1 paciente comprova a eficácia da recomendação docente acoplada à otimização evolutiva: "
        "a sensibilidade aumentou expressivamente, garantindo que praticamente nenhum caso maligno escape da triagem prioritária."
    )
    
    # 5. Integração com LLM e Explicabilidade
    h5 = doc.add_heading("5. Integração com LLM (IA Generativa) e Explicabilidade Clínica", level=1)
    h5.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    
    doc.add_paragraph(
        "Para transpor a barreira entre a matemática do modelo e a tomada de decisão médica, foi implementado o módulo "
        "llm_explainer.py integrado ao OpenAI GPT-4o-mini com Prompt Engineering contextualizado. O sistema recebe a probabilidade "
        "preditiva, o limiar de segurança e os 4 principais atributos apontados pelo SHAP (como deformidade em raio, concavidade e perímetro celular), "
        "emitindo um laudo médico estruturado composto por: 1) Resumo de Risco, 2) Justificativa Biométrica e 3) Conduta Clínica Sugerida."
    )
    
    doc.add_paragraph(
        "Além disso, foi implementado um mecanismo de resiliência (Motor Determinístico de Fallback), assegurando que o sistema opere "
        "e emita laudos completos mesmo em ambientes desconectados ou sem credenciais de API ativas."
    )
    
    # 6. Engenharia de Produção e Interface
    h6 = doc.add_heading("6. Prontidão para Produção e Aplicação Web Interativa", level=1)
    h6.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    
    doc.add_paragraph(
        "A solução conta com uma aplicação interativa em Streamlit (app.py) que permite simular em tempo real diagnósticos clínicos, "
        "exibir o impacto de cada biomarcador e gerar laudos médicos sob demanda. O projeto está 100% containerizado via Dockerfile "
        "e configurado para escalabilidade e deploy em nuvem (Railway/Cloud)."
    )
    
    # 7. Conclusão
    h7 = doc.add_heading("7. Conclusão", level=1)
    h7.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    
    doc.add_paragraph(
        "A Fase 2 consolida a transição de modelos analíticos estáticos para um ecossistema inteligente de apoio à decisão clínica. "
        "A união de Algoritmos Genéticos com foco em Recall e IA Generativa para explicabilidade atendeu rigorosamente ao feedback recebido, "
        "deixando o projeto plenamente estruturado para a futura integração multimodal (prontuários e texto médico) da Fase 3."
    )
    
    output_path = "Relatorio_Entrega_Fase2.docx"
    doc.save(output_path)
    print(f"Relatório gerado com sucesso em: {output_path}")

if __name__ == "__main__":
    create_report()
