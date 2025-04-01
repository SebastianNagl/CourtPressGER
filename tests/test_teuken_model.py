"""
Test zur Überprüfung der Anbindung des Teuken-7B-Modells an die Generierungs-Pipeline.
"""
import os
import pytest
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from courtpressger.generation.pipeline import LLMGenerationPipeline

# Pfad zum lokalen Teuken-Modell
TEUKEN_MODEL_PATH = Path("models/teuken")


class TestTeukenModelIntegration:
    """Tests für die Integration des Teuken-Modells in die Generierungs-Pipeline."""

    def test_model_files_exist(self):
        """Überprüft, ob die Modelldateien existieren."""
        assert TEUKEN_MODEL_PATH.exists(), f"Modellverzeichnis {TEUKEN_MODEL_PATH} nicht gefunden"
        
        # Prüfe auf wichtige Dateien
        config_file = TEUKEN_MODEL_PATH / "config.json"
        assert config_file.exists(), "config.json nicht gefunden"
        
        model_index_file = TEUKEN_MODEL_PATH / "model.safetensors.index.json"
        assert model_index_file.exists(), "model.safetensors.index.json nicht gefunden"
        
        tokenizer_file = TEUKEN_MODEL_PATH / "tokenizer.model"
        assert tokenizer_file.exists(), "tokenizer.model nicht gefunden"
        
        # Prüfe, ob mindestens ein Modell-Shard existiert
        model_shards = list(TEUKEN_MODEL_PATH.glob("model-*-of-*.safetensors"))
        assert len(model_shards) > 0, "Keine Modell-Shards gefunden"

    def test_model_loading(self):
        """Überprüft, ob das Modell geladen werden kann."""
        try:
            # Laden des Tokenizers
            tokenizer = AutoTokenizer.from_pretrained(
                TEUKEN_MODEL_PATH, 
                trust_remote_code=True
            )
            assert tokenizer is not None, "Tokenizer konnte nicht geladen werden"
            
            # Laden der Modellkonfiguration
            config = AutoConfig.from_pretrained(
                TEUKEN_MODEL_PATH,
                trust_remote_code=True
            )
            assert config is not None, "Modellkonfiguration konnte nicht geladen werden"
            
        except Exception as e:
            pytest.fail(f"Fehler beim Laden des Modells: {str(e)}")

    @pytest.mark.parametrize("model_type,name,model_path,api_url", [
        ("local", "teuken-test", str(TEUKEN_MODEL_PATH), "http://localhost:8000/generate"),
    ])
    def test_model_config_in_pipeline(self, model_type, name, model_path, api_url):
        """Überprüft, ob das Modell in der Pipeline konfiguriert werden kann."""
        from courtpressger.evaluation.models import create_model_config
        
        # Erzeuge Modellkonfiguration
        try:
            model_config = create_model_config(
                model_type=model_type,
                name=name,
                model_path=model_path,
                api_url=api_url,
                max_length=512,
                temperature=0.7
            )
            
            assert model_config is not None, "Modellkonfiguration konnte nicht erstellt werden"
            assert model_config["name"] == name, "Falscher Modellname"
            assert callable(model_config["generator_fn"]), "Generator-Funktion ist nicht aufrufbar"
            
        except Exception as e:
            pytest.fail(f"Fehler bei der Modellkonfiguration: {str(e)}")
            
    @pytest.mark.skip(reason="Dieser Test führt eine tatsächliche Generierung durch und ist rechenintensiv")
    def test_model_generation(self):
        """
        Testet die tatsächliche Generierung mit dem Teuken-Modell.
        
        Hinweis: Dieser Test ist mit @pytest.mark.skip markiert, da er rechenintensiv ist
        und nicht bei jedem Test ausgeführt werden sollte.
        """
        from courtpressger.evaluation.models import create_model_config
        import pandas as pd
        
        # Testdaten
        test_data = {
            "ruling": ["Ein Gericht hat entschieden, dass..."],
            "synthetic_prompt": ["Fasse das folgende Urteil in einer Pressemitteilung zusammen:"],
            "press_release": ["Dies ist eine Beispiel-Pressemitteilung."]
        }
        test_df = pd.DataFrame(test_data)
        
        # Modellkonfiguration
        model_config = create_model_config(
            model_type="local",
            name="teuken-test",
            model_path=str(TEUKEN_MODEL_PATH),
            api_url="http://localhost:8000/generate",
            max_length=512,
            temperature=0.7
        )
        
        # Pipeline initialisieren
        pipeline = LLMGenerationPipeline(
            models=[model_config],
            output_dir="data/test_generation"
        )
        
        # Generierung ausführen
        try:
            results = pipeline.run_generation(
                dataset=test_df,
                prompt_column="synthetic_prompt",
                ruling_column="ruling",
                reference_press_column="press_release",
                batch_size=1,
                checkpoint_freq=1
            )
            
            assert results is not None, "Keine Ergebnisse zurückgegeben"
            assert not results.empty, "Leere Ergebnisse"
            assert "generated_press" in results.columns, "Keine generierte Pressemitteilung gefunden"
            
        except Exception as e:
            pytest.fail(f"Fehler bei der Generierung: {str(e)}") 