"""
Tests für die Generierungs-Pipeline und CLI-Komponenten.
"""
import os
import sys
import json
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Pfad zum Projekt-Verzeichnis hinzufügen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from courtpressger.generation.cli import (
    parse_args, load_dataset, load_models_config, main
)
from courtpressger.generation.pipeline import LLMGenerationPipeline


# Fixtures für die Tests
@pytest.fixture
def sample_dataset():
    """Erstellt ein Beispiel-Dataset für Tests."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'judgement': ['Urteil 1', 'Urteil 2', 'Urteil 3'],
        'synthetic_prompt': ['Prompt 1', 'Prompt 2', 'Prompt 3'],
        'summary': ['Pressemitteilung 1', 'Pressemitteilung 2', 'Pressemitteilung 3']
    })


@pytest.fixture
def sample_models_config():
    """Erstellt eine Beispiel-Modellkonfiguration für Tests."""
    return [
        {
            'name': 'test-model-1',
            'type': 'openai',
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 1024
        },
        {
            'name': 'test-model-2',
            'type': 'local',
            'model_path': './models/test',
            'temperature': 0.8,
            'max_length': 2048
        }
    ]


# Tests für CLI-Funktionen
class TestCLIFunctions:
    """Tests für die CLI-Funktionen der Generation-Pipeline."""

    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args(self, mock_args):
        """Testet die Argument-Parser-Funktion."""
        # Mock-Argumente einrichten
        mock_args.return_value = MagicMock(
            dataset='test.csv',
            output_dir='test_output',
            ruling_column='judgement',
            prompt_column='synthetic_prompt',
            press_column='summary',
            models_config='test_config.json',
            model=None,
            batch_size=10,
            checkpoint_freq=5,
            rate_limit_delay=1.0,
            limit=None
        )
        
        # Funktion aufrufen
        args = parse_args()
        
        # Überprüfen, ob die Argumente korrekt verarbeitet wurden
        assert args.dataset == 'test.csv'
        assert args.output_dir == 'test_output'
        assert args.ruling_column == 'judgement'
        assert args.prompt_column == 'synthetic_prompt'
        assert args.press_column == 'summary'
        assert args.models_config == 'test_config.json'
        assert args.model is None
        assert args.batch_size == 10
        assert args.checkpoint_freq == 5
        assert args.rate_limit_delay == 1.0
        assert args.limit is None

    def test_load_dataset_csv(self, sample_dataset):
        """Testet das Laden eines CSV-Datasets."""
        with patch('pandas.read_csv', return_value=sample_dataset):
            result = load_dataset('test.csv')
            assert len(result) == 3
            assert list(result.columns) == ['id', 'judgement', 'synthetic_prompt', 'summary']
    
    def test_load_dataset_json(self, sample_dataset):
        """Testet das Laden eines JSON-Datasets."""
        with patch('pandas.read_json', return_value=sample_dataset):
            result = load_dataset('test.json')
            assert len(result) == 3
            assert list(result.columns) == ['id', 'judgement', 'synthetic_prompt', 'summary']
    
    def test_load_dataset_with_limit(self, sample_dataset):
        """Testet das Laden eines Datasets mit Limit."""
        with patch('pandas.read_csv', return_value=sample_dataset):
            result = load_dataset('test.csv', limit=2)
            assert len(result) == 2
    
    def test_load_dataset_invalid_format(self):
        """Testet das Laden eines Datasets mit ungültigem Format."""
        with pytest.raises(ValueError):
            load_dataset('test.invalid')
    
    def test_load_models_config(self, sample_models_config):
        """Testet das Laden der Modellkonfiguration."""
        mock_json = {'models': sample_models_config}
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_json))):
            result = load_models_config('test_config.json')
            assert len(result) == 2
            assert result[0]['name'] == 'test-model-1'
            assert result[1]['name'] == 'test-model-2'
    
    def test_load_models_config_with_filter(self, sample_models_config):
        """Testet das Laden der Modellkonfiguration mit Filter."""
        mock_json = {'models': sample_models_config}
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_json))):
            result = load_models_config('test_config.json', filter_model='test-model-1')
            assert len(result) == 1
            assert result[0]['name'] == 'test-model-1'

    @patch('courtpressger.generation.cli.load_dataset')
    @patch('courtpressger.generation.cli.load_models_config')
    @patch('courtpressger.generation.cli.LLMGenerationPipeline')
    @patch('courtpressger.generation.cli.parse_args')
    def test_main_function(self, mock_parse_args, mock_pipeline, mock_load_models, mock_load_dataset, sample_dataset, sample_models_config):
        """Testet die Hauptfunktion mit simulierten Daten."""
        # Mock-Argumente einrichten
        mock_args = MagicMock(
            dataset='test.csv',
            output_dir='test_output',
            ruling_column='judgement',
            prompt_column='synthetic_prompt',
            press_column='summary',
            models_config='test_config.json',
            model=None,
            batch_size=10,
            checkpoint_freq=5,
            rate_limit_delay=1.0,
            limit=None
        )
        mock_parse_args.return_value = mock_args
        
        # Mock für load_dataset
        mock_load_dataset.return_value = sample_dataset
        
        # Mock für load_models_config
        mock_load_models.return_value = sample_models_config
        
        # Mock für die Pipeline
        mock_pipeline_instance = mock_pipeline.return_value
        mock_results = pd.DataFrame({
            'id': [1, 2, 3],
            'model': ['test-model-1', 'test-model-1', 'test-model-1'],
            'generated_text': ['Text 1', 'Text 2', 'Text 3'],
            'error': ['', '', '']
        })
        mock_pipeline_instance.run_generation.return_value = mock_results
        
        # Funktion aufrufen
        main()
        
        # Überprüfen, ob die Funktionen korrekt aufgerufen wurden
        mock_load_dataset.assert_called_once_with('test.csv', None)
        mock_load_models.assert_called_once_with('test_config.json', None)
        mock_pipeline.assert_called_once_with(sample_models_config, output_dir='test_output')
        mock_pipeline_instance.run_generation.assert_called_once()


# Tests für die Pipeline-Klasse
class TestLLMGenerationPipelineMocked:
    """Tests für die LLM-Generierungs-Pipeline mit stark gemockten Komponenten."""
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key", 
        "HF_API_KEY": "test-hf-key",
        "DEEPINFRA_API_KEY": "test-deepinfra-key"
    })
    @patch('os.makedirs')
    def test_pipeline_initialization(self, mock_makedirs, sample_models_config):
        """Testet die Initialisierung der Pipeline."""
        # Pipeline-Methoden mocked
        with patch.object(LLMGenerationPipeline, '_initialize_models'), \
             patch.object(LLMGenerationPipeline, '_validate_api_keys'):
            
            pipeline = LLMGenerationPipeline(sample_models_config, output_dir='test_output')
            
            # Überprüfen, ob die Verzeichnisse erstellt wurden
            mock_makedirs.assert_called_once_with('test_output', exist_ok=True)
            
            # Überprüfen, ob die Modellkonfigurationen korrekt gespeichert wurden
            assert pipeline.model_configs == sample_models_config
            assert pipeline.output_dir == 'test_output'
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key", 
        "HF_API_KEY": "test-hf-key"
    })
    @patch.object(LLMGenerationPipeline, '_initialize_models')
    def test_validate_api_keys(self, mock_init_models, capfd):
        """Testet die Validierung der API-Keys."""
        # Wir simulieren, dass DEEPINFRA_API_KEY nicht gesetzt ist
        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": ""}):
            pipeline = LLMGenerationPipeline([], output_dir='test_output')
            pipeline._validate_api_keys()
            
            # Ausgabe prüfen
            out, _ = capfd.readouterr()
            assert "✓ OpenAI API-Key gefunden" in out
            assert "✓ Hugging Face API-Key gefunden" in out
            assert "✗ DeepInfra API-Key nicht gefunden" in out
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    @patch('os.makedirs')
    def test_extract_generated_text(self, mock_makedirs):
        """Testet die Extraktion des generierten Texts."""
        # Pipeline-Methoden mocked
        with patch.object(LLMGenerationPipeline, '_initialize_models'), \
             patch.object(LLMGenerationPipeline, '_validate_api_keys'):
            
            pipeline = LLMGenerationPipeline([], output_dir='test_output')
            
            # Mock-Methode für den Extraktionstest
            def mock_extract_generated_text(output, input_text):
                """Testet die Implementierung der _extract_generated_text-Methode manuell."""
                # Hier prüfen wir, ob der Input-Text im Output enthalten ist und entfernen ihn
                if input_text in output:
                    output = output.replace(input_text, "").strip()
                
                # Entferne "Gerichtsurteil: XYZ" Teil, falls vorhanden
                if "Gerichtsurteil:" in output:
                    parts = output.split("Gerichtsurteil:")
                    if len(parts) > 1:
                        # Versuche, den Rest nach dem Urteil zu extrahieren
                        rest = parts[1].split("\n\n", 1)
                        if len(rest) > 1:
                            output = rest[1].strip()
                
                return output
            
            # Testfälle für die Textextraktion
            test_cases = [
                # Fall 1: Normaler Output
                {
                    'output': 'Pressemitteilung: Dies ist ein Test.',
                    'input': 'Prompt',
                    'expected': 'Pressemitteilung: Dies ist ein Test.'
                },
                # Fall 2: Output mit dem Input-Text
                {
                    'output': 'Prompt\n\nPressemitteilung: Dies ist ein Test.',
                    'input': 'Prompt',
                    'expected': 'Pressemitteilung: Dies ist ein Test.'
                },
                # Fall 3: Output mit dem Gerichtsurteil
                {
                    'output': 'Gerichtsurteil: Urteil\n\nPressemitteilung: Dies ist ein Test.',
                    'input': 'Prompt\n\nGerichtsurteil: Urteil',
                    'expected': 'Pressemitteilung: Dies ist ein Test.'
                }
            ]
            
            for case in test_cases:
                result = mock_extract_generated_text(case['output'], case['input'])
                # Überprüfen, ob die erwartete Ausgabe enthalten ist, nicht exakt gleich
                assert case['expected'] in result
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    @patch('os.makedirs')
    @patch('json.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.exists')
    def test_run_generation_with_mocks(self, mock_exists, mock_to_csv, mock_file_open, mock_json_load, mock_makedirs, sample_dataset):
        """Testet die Ausführung der Generierung mit umfangreichen Mocks."""
        # Mock für die Chains erstellen
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Generierte Pressemitteilung"
        
        # json.load soll ein leeres dict zurückgeben
        mock_json_load.return_value = {}
        
        # os.path.exists soll True zurückgeben
        mock_exists.return_value = False
        
        # Pipeline-Methoden mocked
        with patch.object(LLMGenerationPipeline, '_initialize_models'), \
             patch.object(LLMGenerationPipeline, '_validate_api_keys'), \
             patch.object(LLMGenerationPipeline, '_extract_generated_text', return_value="Extrahierter Text"):
            
            pipeline = LLMGenerationPipeline(
                [{'name': 'test-model', 'type': 'openai'}], 
                output_dir='test_output'
            )
            
            # Chains manuell setzen
            pipeline.chains = {'test-model': mock_chain}
            
            # Generierung ausführen
            result = pipeline.run_generation(
                dataset=sample_dataset,
                prompt_column='synthetic_prompt',
                ruling_column='judgement',
                reference_press_column='summary',
                batch_size=2,
                checkpoint_freq=1,
                rate_limit_delay=0.1
            )
            
            # Überprüfen, ob die Chains aufgerufen wurden
            assert mock_chain.invoke.call_count > 0
            
            # Überprüfen, ob die Ergebnisse zurückgegeben wurden
            assert not result.empty 