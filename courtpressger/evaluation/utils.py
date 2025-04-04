"""
Hilfsfunktionen für die Evaluierung und Visualisierung von Ergebnissen.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path

def load_evaluation_results(results_dir: str) -> Dict[str, Any]:
    """
    Lädt die Evaluierungsergebnisse aus dem angegebenen Verzeichnis.
    
    Args:
        results_dir: Pfad zum Verzeichnis mit den Evaluierungsergebnissen
        
    Returns:
        Dictionary mit den geladenen Ergebnissen
    """
    summary_path = os.path.join(results_dir, "evaluation_summary.json")
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Zusammenfassungsdatei nicht gefunden: {summary_path}")
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    # Detaillierte Ergebnisse für jedes Modell laden
    detailed_results = {}
    for model_name in summary.keys():
        model_results_path = os.path.join(results_dir, f"{model_name}_results.json")
        if os.path.exists(model_results_path):
            with open(model_results_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                detailed_results[model_name] = model_data.get("detailed_results", {})
    
    return {
        "summary": summary,
        "detailed_results": detailed_results
    }

def results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Konvertiert die Evaluierungsergebnisse in ein DataFrame für einfachere Analyse.
    
    Args:
        results: Ergebnisse aus load_evaluation_results()
        
    Returns:
        DataFrame mit den Evaluierungsergebnissen
    """
    summary = results["summary"]
    models = list(summary.keys())
    
    # Metriken aus dem ersten Modell extrahieren (alle sollten die gleichen haben)
    if not models:
        return pd.DataFrame()
    
    first_model = models[0]
    metrics = [k for k in summary[first_model].keys() if k.startswith("avg_")]
    
    # DataFrame erstellen
    data = []
    for model in models:
        model_summary = summary[model]
        row = {"model": model}
        
        # Metriken hinzufügen
        for metric in metrics:
            if metric in model_summary:
                row[metric.replace("avg_", "")] = model_summary[metric]
        
        # Erfolgsmetriken hinzufügen
        if "successful_generations" in model_summary and "total_samples" in model_summary:
            row["success_rate"] = model_summary["successful_generations"] / model_summary["total_samples"]
        
        data.append(row)
    
    return pd.DataFrame(data)

def visualize_rouge_scores(results_df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """
    Visualisiert die ROUGE-Scores aller Modelle als Balkendiagramm.
    
    Args:
        results_df: DataFrame mit Evaluierungsergebnissen
        output_path: Optional, Pfad zum Speichern der Visualisierung
    """
    if results_df.empty:
        print("Keine Daten zur Visualisierung vorhanden.")
        return
    
    # ROUGE-Metriken für die Visualisierung auswählen
    rouge_metrics = [col for col in results_df.columns if col.endswith("_fmeasure")]
    
    if not rouge_metrics:
        print("Keine ROUGE-Metriken zur Visualisierung vorhanden.")
        return
    
    # Modellnamen für die Beschriftung der x-Achse extrahieren
    models = results_df["model"].tolist()
    
    # Daten für das Diagramm vorbereiten
    data = {metric: results_df[metric].tolist() for metric in rouge_metrics}
    
    # Anzahl der Modelle und Metriken
    n_models = len(models)
    n_metrics = len(rouge_metrics)
    
    # Positionen der Balken berechnen
    bar_width = 0.8 / n_metrics
    index = np.arange(n_models)
    
    # Farbkarte für die verschiedenen Metriken
    colors = plt.cm.viridis(np.linspace(0, 1, n_metrics))
    
    # Balkendiagramm erstellen
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (metric, values) in enumerate(data.items()):
        pos = index + i * bar_width - (n_metrics - 1) * bar_width / 2
        ax.bar(pos, values, bar_width, label=metric.replace("_fmeasure", ""), color=colors[i])
    
    # Diagramm anpassen
    ax.set_xlabel('Modell')
    ax.set_ylabel('F1-Score')
    ax.set_title('ROUGE-Scores nach Modell')
    ax.set_xticks(index)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    
    # Speichern oder Anzeigen der Visualisierung
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisierung gespeichert unter: {output_path}")
    else:
        plt.show()

def visualize_single_metric(results_df: pd.DataFrame, 
                             metric: str,
                             output_path: Optional[str] = None) -> None:
    """
    Visualisiert eine einzelne Metrik für alle Modelle als Balkendiagramm.

    Args:
        results_df: DataFrame mit Evaluierungsergebnissen
        metric: Name der zu visualisierenden Metrik (Spaltenname im DataFrame)
        output_path: Optional, Pfad zum Speichern der Visualisierung
    """
    if results_df.empty:
        print(f"Keine Daten zur Visualisierung für Metrik '{metric}' vorhanden.")
        return

    if metric not in results_df.columns:
        print(f"Metrik '{metric}' nicht in den Daten gefunden.")
        return

    # Modellnamen und Werte für die Metrik extrahieren
    models = results_df["model"].tolist()
    values = results_df[metric].tolist()
    n_models = len(models)

    # Positionen der Balken berechnen
    index = np.arange(n_models)

    # Farbkarte (optional, kann auch eine feste Farbe sein)
    colors = plt.cm.viridis(np.linspace(0.3, 0.7, n_models)) # Etwas variieren

    # Balkendiagramm erstellen
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(index, values, color=colors)

    # Diagramm anpassen
    ax.set_xlabel('Modell')
    ax.set_ylabel(metric.replace("_", " ").title()) # Schönere Achsenbeschriftung
    ax.set_title(f'{metric.replace("_", " ").title()} nach Modell')
    ax.set_xticks(index)
    ax.set_xticklabels(models, rotation=45, ha='right')

    # Y-Achsen-Limit sinnvoll setzen (z.B. 0 bis 1 für die meisten Scores)
    if all(0 <= v <= 1 for v in values):
        ax.set_ylim(0, 1)
    else:
        # Kleinen Puffer hinzufügen
        min_val = min(values) if values else 0
        max_val = max(values) if values else 1
        padding = (max_val - min_val) * 0.1
        ax.set_ylim(max(0, min_val - padding), max_val + padding)

    plt.tight_layout()

    # Speichern oder Anzeigen der Visualisierung
    if output_path:
        try:
            plt.savefig(output_path)
            print(f"Visualisierung für '{metric}' gespeichert unter: {output_path}")
            plt.close(fig) # Figur schließen, um Speicher freizugeben
        except Exception as e:
            print(f"Fehler beim Speichern der Visualisierung für '{metric}': {e}")
            plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def visualize_metric_comparison(results_df: pd.DataFrame, 
                               metrics: List[str],
                               output_path: Optional[str] = None) -> None:
    """
    Erstellt ein Radar-Diagramm, das die verschiedenen Metriken für jedes Modell vergleicht.
    
    Args:
        results_df: DataFrame mit Evaluierungsergebnissen
        metrics: Liste der zu visualisierenden Metriken
        output_path: Optional, Pfad zum Speichern der Visualisierung
    """
    if results_df.empty:
        print("Keine Daten zur Visualisierung vorhanden.")
        return
    
    # Überprüfen, ob alle angegebenen Metriken vorhanden sind
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    if not available_metrics:
        print("Keine der angegebenen Metriken ist in den Daten vorhanden.")
        return
    
    # Modellnamen extrahieren
    models = results_df["model"].tolist()
    n_models = len(models)
    
    # Anzahl der Metriken (Achsen im Radar-Diagramm)
    n_metrics = len(available_metrics)
    
    # Winkel für die Achsen des Radar-Diagramms berechnen
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Schließen des Polygons
    
    # Erweiterung der Metrikliste zum Schließen des Polygons
    metrics_plot = available_metrics + [available_metrics[0]]
    
    # Farbkarte für die verschiedenen Modelle
    colors = plt.cm.viridis(np.linspace(0, 1, n_models))
    
    # Radar-Diagramm erstellen
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, model in enumerate(models):
        # Werte für das aktuelle Modell extrahieren und zum Schließen des Polygons erweitern
        values = results_df.loc[results_df["model"] == model, available_metrics].values.flatten().tolist()
        values += values[:1]
        
        # Radar-Plot für das aktuelle Modell zeichnen
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Diagramm anpassen
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_plot[:-1])
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(0)
    ax.set_title("Vergleich der Metriken nach Modell")
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Speichern oder Anzeigen der Visualisierung
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisierung gespeichert unter: {output_path}")
    else:
        plt.show()

def extract_top_examples(results: Dict[str, Any], 
                         model_name: str, 
                         metric: str, 
                         top_n: int = 5, 
                         highest: bool = True) -> List[Dict[str, Any]]:
    """
    Extrahiert die besten oder schlechtesten Beispiele basierend auf einer Metrik.
    
    Args:
        results: Ergebnisse aus load_evaluation_results()
        model_name: Name des zu analysierenden Modells
        metric: Name der Metrik, nach der sortiert werden soll
        top_n: Anzahl der zurückzugebenden Beispiele
        highest: True für die besten, False für die schlechtesten Beispiele
        
    Returns:
        Liste der Top-Beispiele mit Referenz- und generiertem Text
    """
    if model_name not in results["detailed_results"]:
        raise ValueError(f"Modell {model_name} nicht in den Ergebnissen gefunden")
    
    detailed_results = results["detailed_results"][model_name]
    
    # Nur gültige Einträge ohne Fehler berücksichtigen
    valid_examples = {k: v for k, v in detailed_results.items() if "error" not in v and metric in v}
    
    if not valid_examples:
        return []
    
    # Nach der angegebenen Metrik sortieren
    sorted_examples = sorted(
        valid_examples.items(),
        key=lambda x: x[1][metric],
        reverse=highest
    )
    
    # Top-N Beispiele extrahieren
    top_examples = []
    for idx, (example_id, example_data) in enumerate(sorted_examples[:top_n]):
        top_examples.append({
            "index": idx + 1,
            "example_id": example_id,
            "score": example_data[metric],
            "reference_text": example_data["reference_text"],
            "generated_text": example_data["generated_text"]
        })
    
    return top_examples

def create_report(results_dir: str, output_path: str) -> None:
    """
    Erstellt einen HTML-Bericht mit den Evaluierungsergebnissen und Visualisierungen.
    
    Args:
        results_dir: Pfad zum Verzeichnis mit den Evaluierungsergebnissen
        output_path: Pfad zum Speichern des HTML-Berichts
    """
    # Ergebnisse laden
    results = load_evaluation_results(results_dir)
    results_df = results_to_dataframe(results)
    
    # Ausgabeverzeichnis erstellen
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualisierungen generieren und speichern
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    rouge_path = os.path.join(figures_dir, "rouge_scores.png")
    visualize_rouge_scores(results_df, rouge_path)
    
    # Verfügbare Metriken extrahieren (ohne 'model' und mit 'success_rate')
    available_metrics = [col for col in results_df.columns if col not in ["model"]]
    
    # Radar-Diagramm erstellen
    if available_metrics:
        metrics_path = os.path.join(figures_dir, "metrics_comparison.png")
        visualize_metric_comparison(results_df, available_metrics, metrics_path)

    # Einzelne Metrik-Diagramme erstellen
    single_metric_figure_paths = {}
    for metric in available_metrics:
        # Nur für Metriken, die wahrscheinlich Scores sind (z.B. float-Werte)
        # und nicht z.B. 'success_rate', falls das anders visualisiert werden soll.
        # Hier könnten spezifischere Filter hinzugefügt werden.
        if pd.api.types.is_numeric_dtype(results_df[metric]):
            metric_file_name = f"{metric.replace('_', '-')}_comparison.png"
            metric_output_path = os.path.join(figures_dir, metric_file_name)
            visualize_single_metric(results_df, metric, metric_output_path)
            single_metric_figure_paths[metric] = f"figures/{metric_file_name}"

    # HTML-Bericht erstellen
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "  <title>Evaluierungsbericht für Pressemitteilungsgenerierung</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }",
        "    h1, h2, h3 { color: #333; }",
        "    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
        "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "    th { background-color: #f2f2f2; }",
        "    tr:nth-child(even) { background-color: #f9f9f9; }",
        "    .figure { margin: 20px 0; text-align: center; }",
        "    .figure img { max-width: 100%; height: auto; }",
        "    .example { margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }",
        "    .example pre { white-space: pre-wrap; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Evaluierungsbericht für Pressemitteilungsgenerierung</h1>",
        "  <p>Generiert am: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "</p>",
        
        "  <h2>Zusammenfassung der Ergebnisse</h2>"
    ]
    
    # Tabelle mit den Zusammenfassungsergebnissen
    html_content.extend([
        "  <table>",
        "    <tr>",
        "      <th>Modell</th>"
    ])
    
    # Spaltenüberschriften
    for col in results_df.columns:
        if col != "model":
            html_content.append(f"      <th>{col}</th>")
    
    html_content.append("    </tr>")
    
    # Zeilen mit den Modellergebnissen
    for _, row in results_df.iterrows():
        html_content.append("    <tr>")
        html_content.append(f"      <td>{row['model']}</td>")
        
        for col in results_df.columns:
            if col != "model":
                value = row[col]
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                html_content.append(f"      <td>{formatted_value}</td>")
        
        html_content.append("    </tr>")
    
    html_content.append("  </table>")
    
    # Visualisierungen einfügen
    html_content.extend([
        "  <h2>Visualisierungen</h2>",
        "  <div class='figure'>",
        "    <h3>ROUGE-Scores nach Modell</h3>",
        f"    <img src='figures/rouge_scores.png' alt='ROUGE-Scores'>",
        "  </div>"
    ])
    
    if available_metrics:
        html_content.extend([
            "  <div class='figure'>",
            "    <h3>Vergleich der Metriken nach Modell</h3>",
            f"    <img src='figures/metrics_comparison.png' alt='Metrics Comparison'>",
            "  </div>"
        ])
    
    # Einzelne Metrik-Visualisierungen einfügen
    if single_metric_figure_paths:
        html_content.append("  <h3>Einzelmetriken-Vergleich</h3>")
        for metric, fig_path in single_metric_figure_paths.items():
            # Prüfen, ob die Bilddatei tatsächlich erstellt wurde
            full_fig_path = os.path.join(output_dir, fig_path)
            if os.path.exists(full_fig_path):
                 html_content.extend([
                    "  <div class='figure'>",
                    f"    <h4>{metric.replace('_', ' ').title()}</h4>",
                    f"    <img src='{fig_path}' alt='{metric} Comparison'>",
                    "  </div>"
                ])
            else:
                 print(f"Warnung: Bilddatei für Metrik '{metric}' nicht gefunden unter {full_fig_path}")

    # Beispiele für jedes Modell einfügen
    html_content.append("  <h2>Beispiele</h2>")
    
    for model_name in results["summary"].keys():
        html_content.extend([
            f"  <h3>Modell: {model_name}</h3>",
            "  <h4>Beste Beispiele (nach ROUGE-L F1-Score)</h4>"
        ])
        
        # Beste Beispiele
        best_examples = extract_top_examples(
            results, model_name, "rougeL_fmeasure", top_n=3, highest=True
        )
        
        for example in best_examples:
            html_content.extend([
                f"  <div class='example'>",
                f"    <h5>Beispiel {example['index']} (Score: {example['score']:.4f})</h5>",
                f"    <h6>Referenztext:</h6>",
                f"    <pre>{example['reference_text']}</pre>",
                f"    <h6>Generierter Text:</h6>",
                f"    <pre>{example['generated_text']}</pre>",
                f"  </div>"
            ])
        
        html_content.extend([
            "  <h4>Schlechteste Beispiele (nach ROUGE-L F1-Score)</h4>"
        ])
        
        # Schlechteste Beispiele
        worst_examples = extract_top_examples(
            results, model_name, "rougeL_fmeasure", top_n=3, highest=False
        )
        
        for example in worst_examples:
            html_content.extend([
                f"  <div class='example'>",
                f"    <h5>Beispiel {example['index']} (Score: {example['score']:.4f})</h5>",
                f"    <h6>Referenztext:</h6>",
                f"    <pre>{example['reference_text']}</pre>",
                f"    <h6>Generierter Text:</h6>",
                f"    <pre>{example['generated_text']}</pre>",
                f"  </div>"
            ])
    
    # HTML-Dokument abschließen
    html_content.extend([
        "</body>",
        "</html>"
    ])
    
    # HTML-Bericht speichern
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_content))
    
    print(f"Bericht erstellt und gespeichert unter: {output_path}")

def export_examples_to_csv(results: Dict[str, Any], 
                          output_path: str,
                          model_name: Optional[str] = None,
                          limit: Optional[int] = None) -> None:
    """
    Exportiert Beispiele mit Referenz- und generiertem Text in eine CSV-Datei.
    
    Args:
        results: Ergebnisse aus load_evaluation_results()
        output_path: Pfad zum Speichern der CSV-Datei
        model_name: Optional, Name des zu exportierenden Modells (None für alle)
        limit: Optional, maximale Anzahl der zu exportierenden Beispiele pro Modell
    """
    # Modellnamen bestimmen
    if model_name:
        if model_name not in results["detailed_results"]:
            raise ValueError(f"Modell {model_name} nicht in den Ergebnissen gefunden")
        model_names = [model_name]
    else:
        model_names = list(results["detailed_results"].keys())
    
    # Beispiele sammeln
    examples = []
    
    for model in model_names:
        detailed_results = results["detailed_results"][model]
        valid_examples = {k: v for k, v in detailed_results.items() if "error" not in v}
        
        # Optional begrenzen
        if limit:
            examples_ids = list(valid_examples.keys())[:limit]
        else:
            examples_ids = list(valid_examples.keys())
        
        for example_id in examples_ids:
            example_data = valid_examples[example_id]
            
            # ROUGE-Scores extrahieren
            rouge_scores = {
                k: v for k, v in example_data.items() 
                if k not in ["generated_text", "reference_text"] and not k.startswith("error")
            }
            
            examples.append({
                "model": model,
                "example_id": example_id,
                "reference_text": example_data["reference_text"],
                "generated_text": example_data["generated_text"],
                **rouge_scores
            })
    
    # In Dataframe konvertieren und speichern
    if examples:
        df = pd.DataFrame(examples)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Beispiele exportiert nach: {output_path}")
    else:
        print("Keine Beispiele zum Exportieren gefunden.")

def compute_statistical_significance(results: Dict[str, Any], 
                                   metric: str = "rougeL_fmeasure",
                                   baseline_model: Optional[str] = None) -> pd.DataFrame:
    """
    Berechnet die statistische Signifikanz der Unterschiede zwischen Modellen.
    
    Args:
        results: Ergebnisse aus load_evaluation_results()
        metric: Metrik, für die die Signifikanz berechnet werden soll
        baseline_model: Modell, das als Baseline verwendet werden soll (None für paarweise Vergleiche)
        
    Returns:
        DataFrame mit p-Werten für die paarweisen Vergleiche
    """
    try:
        from scipy import stats
    except ImportError:
        print("Für die Berechnung der statistischen Signifikanz wird scipy benötigt. Installiere es mit 'uv add scipy'.")
        return pd.DataFrame()
    
    # Modellnamen und Werte extrahieren
    model_data = {}
    for model_name, detailed_results in results["detailed_results"].items():
        # Nur gültige Einträge ohne Fehler berücksichtigen
        valid_examples = {k: v for k, v in detailed_results.items() if "error" not in v and metric in v}
        values = [v[metric] for v in valid_examples.values()]
        model_data[model_name] = values
    
    models = list(model_data.keys())
    n_models = len(models)
    
    if n_models < 2:
        print("Mindestens zwei Modelle sind für einen Vergleich erforderlich.")
        return pd.DataFrame()
    
    # Ergebnismatrix initialisieren (p-Werte)
    if baseline_model:
        # Vergleich mit einer Baseline
        if baseline_model not in model_data:
            raise ValueError(f"Baseline-Modell {baseline_model} nicht in den Ergebnissen gefunden")
        
        p_values = {}
        baseline_values = model_data[baseline_model]
        
        for model in models:
            if model != baseline_model:
                # T-Test durchführen
                t_stat, p_value = stats.ttest_ind(baseline_values, model_data[model], equal_var=False)
                p_values[model] = p_value
        
        return pd.DataFrame({"model": list(p_values.keys()), "p_value": list(p_values.values())})
    else:
        # Paarweise Vergleiche
        p_values = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    # T-Test durchführen
                    t_stat, p_value = stats.ttest_ind(model_data[model1], model_data[model2], equal_var=False)
                    p_values[i, j] = p_value
                else:
                    p_values[i, j] = 1.0
        
        return pd.DataFrame(p_values, index=models, columns=models) 