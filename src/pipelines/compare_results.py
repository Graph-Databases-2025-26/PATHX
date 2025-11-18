import subprocess
import pandas as pd
import io
import os
import sys

GALOIS_SCRIPT_PATH = r"scripts\galois_eval.py"
GROUND_TRUTH_ROOT = r"ground"
SQL_SUBMISSIONS_ROOT = r"sql_baseline_results"
NL_SUBMISSIONS_ROOT = r"nl_baseline_results"

PAPER_RESULTS = {
    "NL": {
        "F1-Cell": 0.237, "Cardinality": 0.462, "Tuple Constr.": 0.065, "AVG-Score": 0.254,
        "#Tokens (M)": 0.83, "Avg Time": 120.0
    },
    "SQL": {
        "F1-Cell": 0.431, "Cardinality": 0.659, "Tuple Constr.": 0.351, "AVG-Score": 0.481,
        "#Tokens (M)": 0.33, "Avg Time": 61.4
    }
}

METRICS_ORDER = [
    "F1-Cell", "Cardinality", "Tuple Constr.", "AVG-Score", 
    "#Tokens (M)", "Avg Time"
]


def run_galois_evaluation(submissions_path: str, baseline_name: str) -> dict:
    if not os.path.exists(GALOIS_SCRIPT_PATH):
        print(f"ERREUR: Script GALOIS non trouvé à {GALOIS_SCRIPT_PATH}")
        return {}
    if not os.path.isdir(submissions_path):
        print(f"AVERTISSEMENT: Dossier de soumissions '{baseline_name}' non trouvé à {submissions_path}. Skipping.")
        return {}

    command = [
        "python", GALOIS_SCRIPT_PATH,
        "--ground", GROUND_TRUTH_ROOT,
        "--submissions", submissions_path,
        "--cell-metric", "similarity",
        "--tuple-metric", "similarity",
        "--overall",
        "--format", "csv"
    ]
    
    print(f"-> Évaluation de la baseline {baseline_name}...")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        df = pd.read_csv(io.StringIO(result.stdout))
        
        if df.empty or df.iloc[0]["Dataset"] != "ALL":
             print(f"ERREUR: galois_eval.py n'a pas retourné les résultats agrégés pour {baseline_name}.")
             return {}
            
        all_results = df.iloc[0].to_dict()
        our_results = {}
        
        for metric in ["F1-Cell", "Cardinality", "Tuple Constr.", "AVG-Score"]:
            our_results[metric] = float(all_results.get(metric, 0.0))
        
        tokens_val = all_results.get("#Tokens")
        our_results["#Tokens (M)"] = float(tokens_val) / 1_000_000 if tokens_val is not None else 0.0
        our_results["Avg Time"] = float(all_results.get("Avg Time (s)", 0.0))
        
        print(f"-> Résultats {baseline_name} récupérés avec succès.")
        return our_results

    except subprocess.CalledProcessError as e:
        print(f"\nERREUR: galois_eval.py a échoué pour {baseline_name}. Erreur: {e.stderr}")
        return {}
    except Exception as e:
        print(f"\nERREUR: Erreur inattendue lors du traitement de {baseline_name}: {e}")
        return {}


def generate_report_table(our_nl_results: dict, our_sql_results: dict):
    report_data = []
    
    for metric in METRICS_ORDER:
        paper_nl = PAPER_RESULTS["NL"].get(metric)
        paper_sql = PAPER_RESULTS["SQL"].get(metric)
        our_nl = our_nl_results.get(metric, 0.0)
        our_sql = our_sql_results.get(metric, 0.0)
        
        delta_nl = our_nl - paper_nl if paper_nl is not None and our_nl else None
        delta_sql = our_sql - paper_sql if paper_sql is not None and our_sql else None
        
        def fmt_val(val, num_decimals=3):
            if val is None: return ""
            if metric == "#Tokens (M)" or metric == "Avg Time": num_decimals = 2
            return f"{val:,.{num_decimals}f}"
            
        report_data.append({
            "METRIC": metric,
            "NL (Paper)": fmt_val(paper_nl),
            "SQL (Paper)": fmt_val(paper_sql),
            "NL (Our Result)": fmt_val(our_nl),
            "SQL (Our Result)": fmt_val(our_sql),
            "Δ NL": fmt_val(delta_nl),
            "Δ SQL": fmt_val(delta_sql),
        })
        
    df_report = pd.DataFrame(report_data)

    print("\n" + "="*100)
    print("Rapport d'Évaluation des Baselines NL & SQL (Comparaison Paper vs. Équipe) ")
    print("="*100)
    print(df_report.to_markdown(index=False))
    print("\n" + "="*100)
    print("Le tableau est prêt. Vous pouvez le copier directement dans votre README.md.")
    print("="*100)


if __name__ == "__main__":
    our_nl_results = run_galois_evaluation(NL_SUBMISSIONS_ROOT, "NL")
    
    if not our_nl_results:
        print("\nUtilisation des résultats NL par défaut (0.0) pour la génération du rapport.")
        our_nl_results = {m: 0.0 for m in METRICS_ORDER}
    
    our_sql_results = run_galois_evaluation(SQL_SUBMISSIONS_ROOT, "SQL")
    
    if not our_sql_results:
        print("Utilisation des résultats SQL par défaut (0.0) pour la génération du rapport.")
        our_sql_results = {m: 0.0 for m in METRICS_ORDER}

    generate_report_table(our_nl_results, our_sql_results)