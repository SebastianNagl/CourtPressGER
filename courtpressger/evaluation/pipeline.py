"""
Pipeline zur Evaluierung verschiedener LLMs bei der Generierung von Pressemitteilungen.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Callable, Optional
from rouge_score import rouge_scorer
from tqdm import tqdm

from .metrics import compute_all_metrics, NLTK_AVAILABLE, BERT_SCORE_AVAILABLE


class LLMEvaluationPipeline:
    """
    Pipeline zur Evaluierung der Leistung verschiedener LLMs bei der Generierung
    von Pressemitteilungen aus Gerichtsurteilen mit synthetischen Prompts.
    """

    def __init__(self, models_or_names: List[Dict[str, Any]] | List[str], output_dir: str = "data/evaluation",
                 bert_score_model: Optional[str] = None, lang: str = "de"):
        """
        Initialisiert die Evaluierungspipeline.

        Args:
            models_or_names: Entweder eine Liste der zu evaluierenden Modelle (als Dict mit Namen und Konfiguration)
                              oder eine Liste von Modellnamen (Strings), wenn vorhandene Spalten evaluiert werden.
            output_dir: Verzeichnis zum Speichern der Evaluierungsergebnisse
            bert_score_model: Optional, Name des Modells für BERTScore
            lang: Sprachcode für BERTScore
        """
        self.models_or_names = models_or_names
        self.output_dir = output_dir
        self.results = {}
        self.bert_score_model = bert_score_model
        self.lang = lang

        # Bestimmen, ob wir Modellkonfigurationen oder nur Namen haben
        self.evaluate_existing_columns = isinstance(
            models_or_names[0], str) if models_or_names else False

        # Sicherstellen, dass das Ausgabeverzeichnis existiert
        os.makedirs(output_dir, exist_ok=True)

        # Rouge-Scorer für die Evaluation initialisieren
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def run_evaluation(self, dataset: pd.DataFrame, prompt_column: str, ruling_column: str,
                       reference_press_column: str, batch_size: int = 10,
                       checkpoint_freq: int = 50,
                       source_text_column: Optional[str] = None,
                       enable_factual_consistency: bool = False,
                       enable_llm_as_judge: bool = False) -> Dict[str, Any]:
        """
        Führt die Evaluierung für alle Modelle durch.

        Args:
            dataset: DataFrame mit Urteilen, Prompts und Referenz-Pressemitteilungen
            prompt_column: Name der Spalte mit synthetischen Prompts
            ruling_column: Name der Spalte mit Gerichtsurteilen
            reference_press_column: Name der Spalte mit Referenz-Pressemitteilungen
            batch_size: Anzahl der gleichzeitig zu verarbeitenden Einträge
            checkpoint_freq: Häufigkeit der Checkpoint-Speicherung
            source_text_column: Optional, Name der Spalte mit dem Quelltext für 
                               sachliche Konsistenzmetriken (normalerweise gleich ruling_column)
            enable_factual_consistency: Optional, aktiviert QAGS und FactCC (rechenintensiv)
            enable_llm_as_judge: Optional, aktiviert die LLM-as-a-Judge Bewertung mit Claude Sonnet

        Returns:
            Evaluierungsergebnisse als Dictionary
        """
        all_results = {}

        # Iteriere entweder über Modellkonfigurationen oder Modellnamen
        iterator = self.models_or_names

        for item in iterator:
            if self.evaluate_existing_columns:
                model_name = item  # item ist ein String (Modellname)
            else:
                # item ist ein Dict (Modellkonfiguration)
                model_config = item
                model_name = model_config['name']

            print(f"Evaluiere Modell: {model_name}")

            # Überprüfen, ob die Spalte für das Modell im Datensatz vorhanden ist
            if model_name not in dataset.columns:
                print(
                    f"Warnung: Spalte '{model_name}' nicht im Dataset gefunden. Überspringe dieses Modell.")
                all_results[model_name] = {
                    'error': f"Spalte '{model_name}' nicht im Dataset gefunden"}
                continue  # Nächstes Modell

            # Checkpoint-Datei für dieses Modell
            checkpoint_path = os.path.join(
                self.output_dir, f"{model_name}_checkpoint.json")

            # Bereits vorhandene Ergebnisse laden, falls vorhanden
            existing_results = {}
            processed_indices = set()
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                processed_indices = set(map(int, existing_results.keys()))
                print(
                    f"Lade {len(processed_indices)} bestehende Ergebnisse aus Checkpoint")

            model_results = existing_results.copy()

            # Durchlaufe alle Datensätze, die noch nicht verarbeitet wurden
            unprocessed_indices = [i for i in range(
                len(dataset)) if i not in processed_indices]

            for batch_start in tqdm(range(0, len(unprocessed_indices), batch_size),
                                    desc=f"Evaluiere {model_name}"):
                batch_indices = unprocessed_indices[batch_start:batch_start + batch_size]
                batch_data = dataset.iloc[batch_indices]

                for _, row in batch_data.iterrows():
                    idx = row.name
                    ruling = row[ruling_column]
                    prompt = row[prompt_column]
                    reference = row[reference_press_column]

                    # Generierten Text holen:
                    # Entweder aus der Spalte (wenn evaluate_existing_columns)
                    # Oder durch Generierung (wenn nicht evaluate_existing_columns - dieser Pfad wird aktuell nicht genutzt)
                    try:
                        # Überprüfen, ob die Generierung vorhanden ist
                        generated_press = row[model_name]

                        # Wenn keine Generierung verfügbar, überspringen
                        if pd.isna(generated_press) or not generated_press:
                            model_results[str(idx)] = {
                                'error': 'Kein generierter Text verfügbar'}
                            continue

                        # Hole Quelltext für sachliche Konsistenzmetriken, falls aktiviert
                        source_text = None
                        if enable_factual_consistency and source_text_column and source_text_column in row:
                            source_text = row[source_text_column]

                        # ROUGE-Scores berechnen
                        rouge_result = self.scorer.score(
                            reference, generated_press)

                        # Ergebnis-Dictionary erstellen
                        result_dict = {
                            'prompt': prompt,
                            'reference_text': reference,
                            'generated_text': generated_press,
                            'ruling': ruling,
                            'rouge1_precision': rouge_result['rouge1'].precision,
                            'rouge1_recall': rouge_result['rouge1'].recall,
                            'rouge1_fmeasure': rouge_result['rouge1'].fmeasure,
                            'rouge2_precision': rouge_result['rouge2'].precision,
                            'rouge2_recall': rouge_result['rouge2'].recall,
                            'rouge2_fmeasure': rouge_result['rouge2'].fmeasure,
                            'rougeL_precision': rouge_result['rougeL'].precision,
                            'rougeL_recall': rouge_result['rougeL'].recall,
                            'rougeL_fmeasure': rouge_result['rougeL'].fmeasure
                        }

                        # Zusätzliche Metriken berechnen (BLEU, METEOR, BERTScore)
                        additional_metrics = compute_all_metrics(
                            reference=reference,
                            generated=generated_press,
                            bert_score_model=self.bert_score_model,
                            lang=self.lang,
                            source_text=source_text,
                            enable_factual_consistency=enable_factual_consistency,
                            enable_llm_as_judge=enable_llm_as_judge
                        )

                        # Alle Metriken in das Ergebnis-Dictionary einfügen
                        result_dict.update(additional_metrics)

                        model_results[str(idx)] = result_dict

                    except Exception as e:
                        print(
                            f"Fehler bei der Verarbeitung von Index {idx}: {str(e)}")
                        model_results[str(idx)] = {'error': str(e)}

                # Regelmäßig Checkpoints speichern
                if (batch_start + batch_size) % checkpoint_freq == 0:
                    with open(checkpoint_path, 'w', encoding='utf-8') as f:
                        json.dump(model_results, f,
                                  ensure_ascii=False, indent=2)

            # Finalen Checkpoint speichern
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(model_results, f, ensure_ascii=False, indent=2)

            # Zusammenfassung berechnen
            summary = self._compute_summary(model_results)

            # Vollständige Ergebnisse speichern
            with open(os.path.join(self.output_dir, f"{model_name}_results.json"), 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': summary,
                    'detailed_results': model_results
                }, f, ensure_ascii=False, indent=2)

            all_results[model_name] = summary

        # Gesamtergebnisse speichern
        with open(os.path.join(self.output_dir, "evaluation_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        return all_results

    def _compute_summary(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Berechnet zusammenfassende Statistiken aus den Evaluierungsergebnissen.

        Args:
            results: Detaillierte Evaluierungsergebnisse

        Returns:
            Zusammenfassung der Ergebnisse
        """
        # Nur Einträge ohne Fehler berücksichtigen
        valid_results = {k: v for k, v in results.items() if 'error' not in v}

        if not valid_results:
            return {
                'error': 'Keine gültigen Ergebnisse gefunden'
            }

        # Alle metrischen Schlüssel extrahieren, die wir zusammenfassen möchten
        # 1. ROUGE-Metriken
        rouge_metrics = [
            'rouge1_precision', 'rouge1_recall', 'rouge1_fmeasure',
            'rouge2_precision', 'rouge2_recall', 'rouge2_fmeasure',
            'rougeL_precision', 'rougeL_recall', 'rougeL_fmeasure'
        ]

        # 2. BLEU-Metriken
        bleu_metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4']

        # 3. METEOR-Metrik
        meteor_metrics = ['meteor']

        # 4. BERTScore-Metriken
        bertscore_metrics = ['bertscore_precision',
                             'bertscore_recall', 'bertscore_f1']

        # 5. Andere Metriken
        other_metrics = ['keyword_overlap', 'entity_overlap',
                         'length_ratio', 'semantic_similarity']

        # 6. Sachliche Konsistenzmetriken
        factual_consistency_metrics = [
            'qags_score', 'qags_question_count', 'qags_valid_answers',
            'factcc_score', 'factcc_consistency_ratio', 'factcc_claim_count'
        ]

        # 7. LLM-as-a-Judge Metriken
        llm_judge_metrics = [
            'llm_judge_faktische_korrektheit', 'llm_judge_vollständigkeit',
            'llm_judge_klarheit', 'llm_judge_struktur', 'llm_judge_vergleich_mit_referenz',
            'llm_judge_gesamtscore'
        ]

        # Alle Metriken kombinieren
        all_metrics = rouge_metrics + bleu_metrics + meteor_metrics + bertscore_metrics + \
            other_metrics + factual_consistency_metrics + llm_judge_metrics

        # Durchschnittliche Werte für alle Metriken berechnen
        summary = {}

        for metric in all_metrics:
            values = [v[metric] for v in valid_results.values() if metric in v]
            if values:
                summary[f'avg_{metric}'] = sum(values) / len(values)

        # Anzahl der erfolgreichen und fehlgeschlagenen Generierungen
        summary['total_samples'] = len(results)
        summary['successful_generations'] = len(valid_results)
        summary['failed_generations'] = len(results) - len(valid_results)

        return summary
