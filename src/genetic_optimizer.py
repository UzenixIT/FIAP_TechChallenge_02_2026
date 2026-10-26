import random
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
from .model_trainer import build_logistic_pipeline, load_and_prep_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_individual(individual, X_train, y_train):
    """
    Função Fitness do Algoritmo Genético.
    individual: [C, class_weight_idx, threshold]
    Genes:
      - C (float): 0.01 a 20.0
      - class_weight_idx (int): 0 (None) ou 1 ('balanced')
      - threshold (float): 0.15 a 0.60 (Limiar de decisão clínico, atendendo ao feedback do professor)
    
    Fitness:
      Maximiza primariamente o Recall (peso 0.8) e secundariamente o F1-Score (peso 0.2),
      penalizando severamente qualquer modelo que reduza a segurança clínica.
    """
    c_val = max(0.01, float(individual[0]))
    cw_idx = int(round(individual[1]))
    class_weight = 'balanced' if cw_idx == 1 else None
    threshold = float(np.clip(individual[2], 0.10, 0.70))
    
    try:
        pipeline = build_logistic_pipeline(C=c_val, class_weight=class_weight)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
        
        y_pred = (y_proba >= threshold).astype(int)
        
        rec = recall_score(y_train, y_pred, zero_division=0)
        f1 = f1_score(y_train, y_pred, zero_division=0)
        
        # Fitness composta clínica: 80% Recall (zero falso negativo) + 20% F1-score
        fitness = 0.80 * rec + 0.20 * f1
        return (fitness, )
    except Exception as e:
        logging.error(f"Erro na avaliacao do individuo: {e}")
        return (0.0, )

def setup_deap_toolbox(X_train, y_train, mutation_sigma=0.5):
    """Configura o ambiente DEAP com operadores genéticos."""
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual
        
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # 0: C (float: 0.05 a 15.0)
    # 1: class_weight (int: 0 ou 1)
    # 2: threshold clinico (float: 0.15 a 0.55)
    toolbox.register("attr_c", random.uniform, 0.05, 15.0)
    toolbox.register("attr_cw", random.randint, 0, 1)
    toolbox.register("attr_thresh", random.uniform, 0.15, 0.55)
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_c, toolbox.attr_cw, toolbox.attr_thresh), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_individual, X_train=X_train, y_train=y_train)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=mutation_sigma, indpb=0.25)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def run_experiment(exp_name, pop_size, ngen, cxpb, mutpb, mutation_sigma, X_train, y_train):
    """Executa um experimento com parâmetros específicos do AG."""
    logging.info(f"--- Iniciando {exp_name}: Pop={pop_size}, Gen={ngen}, cxpb={cxpb}, mutpb={mutpb} ---")
    toolbox = setup_deap_toolbox(X_train, y_train, mutation_sigma=mutation_sigma)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, 
                                       stats=stats, halloffame=hof, verbose=False)
    
    best = hof[0]
    best_c = max(0.01, float(best[0]))
    best_cw = 'balanced' if int(round(best[1])) == 1 else None
    best_thresh = float(np.clip(best[2], 0.10, 0.70))
    best_fitness = best.fitness.values[0]
    
    logging.info(f"Resultado {exp_name} -> Fitness: {best_fitness:.4f} | C: {best_c:.4f} | CW: {best_cw} | Thresh: {best_thresh:.4f}")
    
    return {
        "exp_name": exp_name,
        "pop_size": pop_size,
        "ngen": ngen,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "best_individual": [best_c, best_cw, best_thresh],
        "best_fitness": best_fitness,
        "logbook": logbook
    }

def run_all_experiments():
    """Executa os 3 experimentos obrigatórios exigidos pelo Tech Challenge Fase 2."""
    X_train, X_test, y_train, y_test = load_and_prep_data("data/data.csv")
    
    configs = [
        {
            "name": "Experimento 1 (Conservador)",
            "pop_size": 20, "ngen": 10, "cxpb": 0.5, "mutpb": 0.2, "sigma": 0.3
        },
        {
            "name": "Experimento 2 (Alta Exploracao)",
            "pop_size": 35, "ngen": 15, "cxpb": 0.7, "mutpb": 0.4, "sigma": 0.8
        },
        {
            "name": "Experimento 3 (Populacao Expandida & Foco Clinico)",
            "pop_size": 50, "ngen": 20, "cxpb": 0.6, "mutpb": 0.25, "sigma": 0.5
        }
    ]
    
    results = []
    for cfg in configs:
        res = run_experiment(
            cfg["name"], 
            pop_size=cfg["pop_size"], 
            ngen=cfg["ngen"], 
            cxpb=cfg["cxpb"], 
            mutpb=cfg["mutpb"], 
            mutation_sigma=cfg["sigma"],
            X_train=X_train, 
            y_train=y_train
        )
        results.append(res)
        
    # Selecionar o melhor entre os 3 experimentos
    best_exp = max(results, key=lambda x: x["best_fitness"])
    best_c, best_cw, best_thresh = best_exp["best_individual"]
    
    print("\n" + "="*70)
    print("[FINAL] RESULTADO DA OTIMIZACAO GENETICA (FASE 2)")
    print(f"Melhor Configuracao: {best_exp['exp_name']}")
    print(f"Parametros: C={best_c:.4f}, class_weight={best_cw}, Limiar Clinico={best_thresh:.4f}")
    print(f"Melhor Fitness Composta (CV): {best_exp['best_fitness']:.4f}")
    print("="*70 + "\n")
    
    # 1. Avaliar Modelo Original (Baseline Fase 1: C=1.0, CW=None, Thresh=0.50)
    baseline_pipeline = build_logistic_pipeline(C=1.0, class_weight=None)
    baseline_pipeline.fit(X_train, y_train)
    y_test_base_proba = baseline_pipeline.predict_proba(X_test)[:, 1]
    y_test_base_pred = (y_test_base_proba >= 0.50).astype(int)
    
    base_rec = recall_score(y_test, y_test_base_pred)
    base_prec = precision_score(y_test, y_test_base_pred)
    base_f1 = f1_score(y_test, y_test_base_pred)
    base_acc = accuracy_score(y_test, y_test_base_pred)
    cm_base = confusion_matrix(y_test, y_test_base_pred)
    
    # 2. Treinar e Avaliar Modelo Otimizado com Algoritmo Genético + Threshold
    opt_pipeline = build_logistic_pipeline(C=best_c, class_weight=best_cw)
    opt_pipeline.fit(X_train, y_train)
    y_test_opt_proba = opt_pipeline.predict_proba(X_test)[:, 1]
    y_test_opt_pred = (y_test_opt_proba >= best_thresh).astype(int)
    
    opt_rec = recall_score(y_test, y_test_opt_pred)
    opt_prec = precision_score(y_test, y_test_opt_pred)
    opt_f1 = f1_score(y_test, y_test_opt_pred)
    opt_acc = accuracy_score(y_test, y_test_opt_pred)
    cm_opt = confusion_matrix(y_test, y_test_opt_pred)
    
    # Comparativo
    comparison_df = pd.DataFrame({
        "Métrica": ["Recall (Maligno)", "Precisão", "F1-Score", "Acurácia", "Falsos Negativos (FN)"],
        "Modelo Fase 1 (Original, T=0.5)": [
            f"{base_rec*100:.2f}%", f"{base_prec*100:.2f}%", f"{base_f1*100:.2f}%", f"{base_acc*100:.2f}%", cm_base[1, 0]
        ],
        "Modelo Fase 2 (AG + Threshold Clínico)": [
            f"{opt_rec*100:.2f}%", f"{opt_prec*100:.2f}%", f"{opt_f1*100:.2f}%", f"{opt_acc*100:.2f}%", cm_opt[1, 0]
        ]
    })
    
    print("\n--- COMPARATIVO TESTE CEGO: FASE 1 vs FASE 2 ---")
    print(comparison_df.to_string(index=False))
    
    # Salvar artefatos
    joblib.dump({
        "pipeline": opt_pipeline,
        "best_threshold": best_thresh,
        "best_c": best_c,
        "best_cw": best_cw,
        "feature_names": list(X_train.columns)
    }, "data/best_model_phase2.joblib")
    
    # Salvar resultados e logbooks para o notebook
    joblib.dump({
        "experiments": results,
        "comparison_df": comparison_df,
        "best_exp": best_exp,
        "cm_base": cm_base,
        "cm_opt": cm_opt
    }, "data/experiments_results.joblib")
    
    logging.info("Modelo otimizado e resultados salvos com sucesso em data/!")
    return results, comparison_df

if __name__ == "__main__":
    run_all_experiments()
