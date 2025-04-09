import pytest
import pandas as pd
import os
import json
from unittest.mock import patch, MagicMock, mock_open

from courtpressger.evaluation.pipeline import LLMEvaluationPipeline

# Definiere ein Test-Fixture für Beispieldaten
@pytest.fixture
def sample_dataset():
    data = {
        'ruling': ['Urteil 1 Text', 'Urteil 2 Text'],
        'prompt': ['Prompt 1', 'Prompt 2'],
        'reference_press': ['Referenz PM 1', 'Referenz PM 2'],
        'model_a': ['Generierte PM 1 von A', 'Generierte PM 2 von A'],
        'model_b': ['Generierte PM 1 von B', 'Generierte PM 2 von B mit Fehler'], # Beispiel mit fehlerhaftem Eintrag
        'model_c': [None, 'Generierte PM 2 von C'] # Beispiel mit fehlendem Eintrag
    }
    return pd.DataFrame(data)

# Definiere ein Test-Fixture für die Pipeline-Instanz
@pytest.fixture
def evaluation_pipeline(tmp_path):
    output_dir = tmp_path / "evaluation_output"
    models_to_evaluate = ['model_a', 'model_b', 'model_c', 'model_d_missing'] # model_d fehlt im Dataset
    pipeline = LLMEvaluationPipeline(models_or_names=models_to_evaluate, output_dir=str(output_dir))
    return pipeline, output_dir

# Testklasse für die LLMEvaluationPipeline
class TestLLMEvaluationPipeline:

    def test_init(self, evaluation_pipeline):
        """Testet die Initialisierung der Pipeline."""
        pipeline, output_dir = evaluation_pipeline
        assert pipeline.evaluate_existing_columns is True
        assert pipeline.output_dir == str(output_dir)
        assert os.path.exists(output_dir) # Prüfen, ob das Ausgabeverzeichnis erstellt wurde
        assert isinstance(pipeline.models_or_names, list)
        assert pipeline.models_or_names == ['model_a', 'model_b', 'model_c', 'model_d_missing']

    @patch('courtpressger.evaluation.pipeline.rouge_scorer.RougeScorer')
    @patch('courtpressger.evaluation.pipeline.compute_all_metrics')
    @patch('builtins.open', new_callable=mock_open)
    def test_run_evaluation_existing_columns(self, mock_file_open, mock_compute_all, mock_rouge_scorer, 
                                             evaluation_pipeline, sample_dataset, tmp_path):
        """Testet die run_evaluation Methode, wenn Spaltennamen übergeben werden."""
        pipeline, output_dir = evaluation_pipeline
        
        # Mock ROUGE Scorer
        mock_rouge_instance = MagicMock()
        mock_rouge_instance.score.return_value = {
            'rouge1': MagicMock(precision=0.8, recall=0.7, fmeasure=0.75),
            'rouge2': MagicMock(precision=0.6, recall=0.5, fmeasure=0.55),
            'rougeL': MagicMock(precision=0.7, recall=0.6, fmeasure=0.65)
        }
        mock_rouge_scorer.return_value = mock_rouge_instance
        
        # Mock compute_all_metrics
        mock_compute_all.return_value = {
            'bleu1': 0.7, 'bleu2': 0.6, 'bleu3': 0.5, 'bleu4': 0.4,
            'meteor': 0.65,
            'bertscore_precision': 0.9, 'bertscore_recall': 0.85, 'bertscore_f1': 0.87,
            'keyword_overlap': 0.5, 'entity_overlap': 0.4, 'length_ratio': 1.1, 'semantic_similarity': 0.78,
            # LLMAsJudge-spezifische Metriken
            'llm_judge_faktische_korrektheit': 8, 
            'llm_judge_vollständigkeit': 7,
            'llm_judge_klarheit': 9,
            'llm_judge_struktur': 8,
            'llm_judge_vergleich_mit_referenz': 8,
            'llm_judge_gesamtscore': 8.0
        }

        # Mock os.path.exists für Checkpoints (simuliert keinen existierenden Checkpoint)
        with patch('os.path.exists', return_value=False):
            results = pipeline.run_evaluation(
                dataset=sample_dataset,
                prompt_column='prompt',
                ruling_column='ruling',
                reference_press_column='reference_press',
                enable_llm_as_judge=True  # LLM-as-a-Judge aktivieren
            )

        # --- Assertions ---
        assert 'model_a' in results
        assert 'model_b' in results
        assert 'model_c' in results # Modell C wird evaluiert, aber mit Fehlern für den ersten Eintrag
        assert 'model_d_missing' in results # Modell D wird übersprungen
        assert 'error' in results['model_d_missing']
        assert results['model_d_missing']['error'] == "Spalte 'model_d_missing' nicht im Dataset gefunden"
        
        # Prüfen, ob ROUGE und compute_all_metrics aufgerufen wurden (für gültige Einträge)
        # Model A hat 2 gültige Einträge
        # Model B hat 1 gültigen Eintrag (der zweite ist Text, aber simuliert als "fehlerhaft") -> hier müssen wir prüfen, wie Fehler behandelt werden
        # Model C hat 1 gültigen Eintrag (der erste ist None)
        # Erwartete Aufrufe für score: 2 (model_a) + 1 (model_b) + 1 (model_c) = 4
        # Erwartete Aufrufe für compute_all: 2 (model_a) + 1 (model_b) + 1 (model_c) = 4
        assert mock_rouge_instance.score.call_count == 4 
        assert mock_compute_all.call_count == 4

        # Prüfen, ob Checkpoint- und Ergebnisdateien geschrieben wurden
        expected_checkpoint_calls = [
            f'{output_dir}/model_a_checkpoint.json',
            f'{output_dir}/model_b_checkpoint.json',
            f'{output_dir}/model_c_checkpoint.json',
            # model_d_missing hat keinen Checkpoint
        ]
        expected_results_calls = [
            f'{output_dir}/model_a_results.json',
            f'{output_dir}/model_b_results.json',
            f'{output_dir}/model_c_results.json',
            f'{output_dir}/evaluation_summary.json' # Die Gesamtübersicht
        ]
        
        # Überprüfen der Schreibzugriffe
        write_calls = [call_args[0][0] for call_args in mock_file_open.call_args_list if call_args[0][1] == 'w']
        
        # Jeder Checkpoint wird initial (leer?) und final geschrieben
        for fname in expected_checkpoint_calls:
             # Jeder Checkpoint wird mindestens einmal geschrieben (final)
            assert fname in write_calls 
        for fname in expected_results_calls:
            assert fname in write_calls

        # Prüfen der Summary-Struktur (Beispiel für model_a)
        summary_a = results['model_a']
        assert 'avg_rouge1_fmeasure' in summary_a
        assert 'avg_bleu4' in summary_a
        assert 'avg_bertscore_f1' in summary_a
        assert 'avg_llm_judge_gesamtscore' in summary_a  # LLM-as-a-Judge Gesamtscore
        assert 'avg_llm_judge_faktische_korrektheit' in summary_a  # Einzelne LLM-as-a-Judge Metriken
        assert summary_a['successful_generations'] == 2 # Model A hat 2 erfolgreiche Generierungen
        assert summary_a['failed_generations'] == 0

        # Prüfen der Summary für Model C (mit einem Fehler)
        summary_c = results['model_c']
        assert summary_c['successful_generations'] == 1 # Nur der zweite Eintrag ist gültig
        assert summary_c['failed_generations'] == 1 # Der erste Eintrag war None

    def test_compute_summary_no_valid_results(self, evaluation_pipeline):
        """Testet die _compute_summary Methode, wenn keine gültigen Ergebnisse vorhanden sind."""
        pipeline, _ = evaluation_pipeline
        results_with_errors = {
            '0': {'error': 'Some error'},
            '1': {'error': 'Another error'}
        }
        summary = pipeline._compute_summary(results_with_errors)
        assert 'error' in summary
        assert summary['error'] == 'Keine gültigen Ergebnisse gefunden'

    def test_compute_summary_with_results(self, evaluation_pipeline):
        """Testet die _compute_summary Methode mit gültigen Ergebnissen."""
        pipeline, _ = evaluation_pipeline
        # Beispielhafte detaillierte Ergebnisse
        detailed_results = {
            '0': {
                'generated_text': 'gen 1', 'reference_text': 'ref 1',
                'rouge1_fmeasure': 0.8, 'rouge2_fmeasure': 0.7, 'rougeL_fmeasure': 0.75,
                'bleu4': 0.6, 'meteor': 0.65, 'bertscore_f1': 0.85,
                'keyword_overlap': 0.5, 'entity_overlap': 0.4, 'length_ratio': 1.1, 'semantic_similarity': 0.78,
                # LLM-as-a-Judge Metriken
                'llm_judge_faktische_korrektheit': 8,
                'llm_judge_vollständigkeit': 7,
                'llm_judge_klarheit': 9,
                'llm_judge_struktur': 8,
                'llm_judge_vergleich_mit_referenz': 8,
                'llm_judge_gesamtscore': 8.0
            },
            '1': {
                'generated_text': 'gen 2', 'reference_text': 'ref 2',
                'rouge1_fmeasure': 0.7, 'rouge2_fmeasure': 0.6, 'rougeL_fmeasure': 0.65,
                'bleu4': 0.5, 'meteor': 0.55, 'bertscore_f1': 0.75,
                'keyword_overlap': 0.4, 'entity_overlap': 0.3, 'length_ratio': 1.0, 'semantic_similarity': 0.68,
                # LLM-as-a-Judge Metriken
                'llm_judge_faktische_korrektheit': 7,
                'llm_judge_vollständigkeit': 6,
                'llm_judge_klarheit': 8,
                'llm_judge_struktur': 7,
                'llm_judge_vergleich_mit_referenz': 7,
                'llm_judge_gesamtscore': 7.0
            },
            '2': {'error': 'Failed generation'} # Eintrag mit Fehler
        }
        summary = pipeline._compute_summary(detailed_results)
        
        # Prüfen der Zusammenfassungsberechnung
        assert summary['total_samples'] == 3
        assert summary['successful_generations'] == 2
        assert summary['failed_generations'] == 1
        
        # Prüfen einiger Durchschnittswerte (Beispiele)
        assert summary['avg_rouge1_fmeasure'] == pytest.approx((0.8 + 0.7) / 2)
        assert summary['avg_bleu4'] == pytest.approx((0.6 + 0.5) / 2)
        assert summary['avg_bertscore_f1'] == pytest.approx((0.85 + 0.75) / 2)
        assert 'avg_rouge1_precision' not in summary # Nicht im Beispiel-Result-Dict enthalten
        assert summary['avg_llm_judge_gesamtscore'] == pytest.approx((8.0 + 7.0) / 2)
        assert summary['avg_llm_judge_faktische_korrektheit'] == pytest.approx((8 + 7) / 2)
        assert summary['avg_llm_judge_vollständigkeit'] == pytest.approx((7 + 6) / 2)

    # Hilfsfunktion für mock_open side_effect im Checkpoint-Test
    def _mock_open_side_effect(self, checkpoint_path, checkpoint_content):
        def side_effect(path, mode='r', **kwargs):
            if path == checkpoint_path and 'r' in mode:
                # Simulieren des Lesens des Checkpoints
                m = mock_open(read_data=json.dumps(checkpoint_content))
                return m.return_value
            else:
                # Standardverhalten für andere Dateizugriffe (z.B. Schreiben)
                return mock_open().return_value
        return side_effect

    @patch('courtpressger.evaluation.pipeline.rouge_scorer.RougeScorer')
    @patch('courtpressger.evaluation.pipeline.compute_all_metrics')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists') # Mock os.path.exists für Checkpoints
    def test_run_evaluation_with_checkpoint(self, mock_exists, mock_file_open, mock_compute_all, mock_rouge_scorer, 
                                            evaluation_pipeline, sample_dataset, tmp_path):
        """Testet die run_evaluation Methode mit Laden eines Checkpoints."""
        pipeline, output_dir = evaluation_pipeline
        model_name = 'model_a'
        checkpoint_path = os.path.join(output_dir, f"{model_name}_checkpoint.json")

        # Mock ROUGE & Metrics wie im anderen Test
        mock_rouge_instance = MagicMock()
        mock_rouge_instance.score.return_value = {'rouge1': MagicMock(fmeasure=0.75), 'rouge2': MagicMock(fmeasure=0.55), 'rougeL': MagicMock(fmeasure=0.65)}
        mock_rouge_scorer.return_value = mock_rouge_instance
        mock_compute_all.return_value = {'bleu4': 0.6, 'bertscore_f1': 0.85}

        # Simulieren, dass der Checkpoint für model_a existiert und einen Eintrag enthält
        mock_exists.side_effect = lambda path: path == checkpoint_path
        
        # Inhalt des simulierten Checkpoints
        checkpoint_content = {
            "0": { # Index 0 wurde bereits verarbeitet
                'generated_text': 'Generierte PM 1 von A', 'reference_text': 'Referenz PM 1', 
                'rouge1_fmeasure': 0.8, 'rouge2_fmeasure': 0.7, 'rougeL_fmeasure': 0.75, 
                'bleu4': 0.6, 'bertscore_f1': 0.85,
                # ... andere Metriken ...
            }
        }
        # Konfigurieren von mock_open mit der Hilfsfunktion
        mock_file_open.side_effect = self._mock_open_side_effect(checkpoint_path, checkpoint_content)


        results = pipeline.run_evaluation(
            dataset=sample_dataset,
            prompt_column='prompt',
            ruling_column='ruling',
            reference_press_column='reference_press'
        )
        
        # --- Assertions ---
        # Nur der zweite Eintrag (Index 1) sollte neu berechnet werden
        # ROUGE sollte nur 1x für den neuen Eintrag von model_a aufgerufen werden
        # Plus 1x für model_b (gültig), 1x für model_c (gültig) = 3 Aufrufe insgesamt
        assert mock_rouge_instance.score.call_count == 3 
        assert mock_compute_all.call_count == 3
        
        # Prüfen, ob der Checkpoint gelesen wurde
        read_calls = [call_args[0][0] for call_args in mock_file_open.call_args_list if call_args[0][1] == 'r']
        assert checkpoint_path in read_calls

        # Die Summary für model_a sollte beide Einträge berücksichtigen (den aus dem Checkpoint und den neu berechneten)
        summary_a = results['model_a']
        assert summary_a['successful_generations'] == 2
        assert summary_a['failed_generations'] == 0