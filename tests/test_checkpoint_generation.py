# Diese Datei enthält die aktuellen Verbesserungen für das Checkpoint-Speichern

"""
Tests für die Checkpoint-Erstellung in der Generator-Funktionalität.
"""

import os
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from courtpressger.synthetic_prompts.generator import process_batch, generate_synthetic_prompt


class MockAnthropicResponse:
    """Mock für die Anthropic API Response."""
    
    def __init__(self, content):
        self.content = [MagicMock(text=content)]
        self.usage = MagicMock(input_tokens=100, output_tokens=50)


class MockAnthropicClient:
    """Mock für den Anthropic Client."""
    
    def __init__(self, response_text="Mock synthetischer Prompt"):
        self.response_text = response_text
        self.messages = MagicMock()
        self.messages.create = MagicMock(return_value=MockAnthropicResponse(response_text))


@pytest.fixture
def sample_dataframe():
    """Erstellt einen Beispiel-DataFrame für Tests."""
    data = {
        'judgement': [f"Gerichtsurteil {i}" for i in range(20)],
        'summary': [f"Pressemitteilung {i}" for i in range(20)]
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_client():
    """Erstellt einen Mock-Anthropic-Client."""
    return MockAnthropicClient()


@pytest.fixture
def mock_generate_synthetic_prompt():
    """
    Patcht die generate_synthetic_prompt-Funktion, damit keine echten API-Aufrufe erfolgen.
    """
    with patch('courtpressger.synthetic_prompts.generator.generate_synthetic_prompt') as mock_func:
        mock_func.return_value = "Mock synthetischer Prompt für Tests"
        yield mock_func


def test_checkpoint_creation(sample_dataframe, mock_client, mock_generate_synthetic_prompt):
    """
    Testet, ob Checkpoints korrekt erstellt werden.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Parameter für den Test
        batch_size = 5
        save_interval = 1  # Speichere nach jedem Batch
        output_prefix = "test_checkpoint"
        checkpoint_dir = Path(temp_dir)
        
        # Prozess ausführen
        process_batch(
            df=sample_dataframe,
            batch_size=batch_size,
            start_idx=0,
            save_interval=save_interval,
            fix_errors=False,
            checkpoint_dir=checkpoint_dir,
            output_prefix=output_prefix,
            client=mock_client
        )
        
        # Überprüfe, ob die erwarteten Checkpoint-Dateien erstellt wurden
        expected_checkpoints = [f"{output_prefix}_{(i+1)*batch_size}.csv" for i in range(4)]
        
        for checkpoint in expected_checkpoints:
            checkpoint_path = checkpoint_dir / checkpoint
            assert checkpoint_path.exists(), f"Checkpoint {checkpoint} wurde nicht erstellt"
            
            # Lade den Checkpoint und überprüfe den Inhalt
            df = pd.read_csv(checkpoint_path)
            assert 'synthetic_prompt' in df.columns
            assert df['synthetic_prompt'].notna().sum() > 0


def test_resume_from_checkpoint(sample_dataframe, mock_client, mock_generate_synthetic_prompt):
    """
    Testet, ob die Wiederaufnahme von einem Checkpoint korrekt funktioniert.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Parameter für den Test
        batch_size = 5
        save_interval = 1
        output_prefix = "test_checkpoint"
        checkpoint_dir = Path(temp_dir)
        
        # Ersten Durchlauf mit 10 Einträgen (2 Batches)
        process_batch(
            df=sample_dataframe,
            batch_size=batch_size,
            start_idx=0,
            save_interval=save_interval,
            fix_errors=False,
            checkpoint_dir=checkpoint_dir,
            output_prefix=output_prefix,
            client=mock_client
        )
        
        # Zweiten Durchlauf ab Index 10 starten
        process_batch(
            df=sample_dataframe,
            batch_size=batch_size,
            start_idx=10,  # Starte von Index 10
            save_interval=save_interval,
            fix_errors=False,
            checkpoint_dir=checkpoint_dir,
            output_prefix=output_prefix,
            client=mock_client
        )
        
        # Überprüfe, ob alle erwarteten Checkpoint-Dateien erstellt wurden
        all_checkpoints = list(checkpoint_dir.glob(f"{output_prefix}_*.csv"))
        assert len(all_checkpoints) >= 4, "Nicht alle erwarteten Checkpoints wurden erstellt"


def test_output_prefix_override(sample_dataframe, mock_client, mock_generate_synthetic_prompt):
    """
    Testet, ob der output_prefix nicht überschrieben wird, wenn er explizit übergeben wird.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Parameter für den Test
        batch_size = 5
        save_interval = 1
        custom_prefix = "custom_prefix"  # Benutzerdefinierter Präfix
        default_prefix = "synthetic_prompts_checkpoint"  # Standard-Präfix
        checkpoint_dir = Path(temp_dir)
        
        # Ausführen mit benutzerdefiniertem Präfix
        process_batch(
            df=sample_dataframe,
            batch_size=batch_size,
            start_idx=0,
            save_interval=save_interval,
            fix_errors=False,
            checkpoint_dir=checkpoint_dir,
            output_prefix=custom_prefix,
            client=mock_client
        )
        
        # Überprüfe, ob Checkpoints mit dem benutzerdefinierten Präfix erstellt wurden
        custom_checkpoints = list(checkpoint_dir.glob(f"{custom_prefix}_*.csv"))
        default_checkpoints = list(checkpoint_dir.glob(f"{default_prefix}_*.csv"))
        
        assert len(custom_checkpoints) > 0, "Keine Checkpoints mit benutzerdefiniertem Präfix erstellt"
        assert len(default_checkpoints) == 0, "Es wurden fälschlicherweise Checkpoints mit Standard-Präfix erstellt"


def test_checkpoint_creation_batch_size_10(sample_dataframe, mock_client, mock_generate_synthetic_prompt):
    """
    Testet, ob mit Batch-Größe 10 Checkpoints korrekt erstellt werden.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Parameter für den Test
        batch_size = 10  # Die neue Standardgröße
        save_interval = 1  # Speichere nach jedem Batch
        output_prefix = "test_checkpoint"
        checkpoint_dir = Path(temp_dir)
        
        # Prozess ausführen
        process_batch(
            df=sample_dataframe,
            batch_size=batch_size,
            start_idx=0,
            save_interval=save_interval,
            fix_errors=False,
            checkpoint_dir=checkpoint_dir,
            output_prefix=output_prefix,
            client=mock_client
        )
        
        # Überprüfe, ob die erwarteten Checkpoint-Dateien erstellt wurden
        # Mit Batch-Größe 10 und 20 Einträgen erwarten wir 2 Checkpoints (10, 20)
        expected_checkpoints = [f"{output_prefix}_{(i+1)*batch_size}.csv" for i in range(2)]
        
        for checkpoint in expected_checkpoints:
            checkpoint_path = checkpoint_dir / checkpoint
            assert checkpoint_path.exists(), f"Checkpoint {checkpoint} wurde nicht erstellt"
            
            # Lade den Checkpoint und überprüfe den Inhalt
            df = pd.read_csv(checkpoint_path)
            assert 'synthetic_prompt' in df.columns
            assert df['synthetic_prompt'].notna().sum() > 0


def test_checkpoint_after_each_batch(sample_dataframe, mock_client, mock_generate_synthetic_prompt):
    """
    Testet, ob nach jedem Batch ein Checkpoint erstellt wird (save_interval=1).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Parameter für den Test
        batch_size = 5  # Kleinere Batch-Größe für mehr Checkpoints
        save_interval = 1  # Speichere nach jedem Batch
        output_prefix = "test_checkpoint"
        checkpoint_dir = Path(temp_dir)
        
        # Prozess ausführen
        process_batch(
            df=sample_dataframe,
            batch_size=batch_size,
            start_idx=0,
            save_interval=save_interval,
            fix_errors=False,
            checkpoint_dir=checkpoint_dir,
            output_prefix=output_prefix,
            client=mock_client
        )
        
        # Mit Batchgröße 5 und Dataset von 20 Einträgen erwarten wir 4 Checkpoints
        all_checkpoints = list(checkpoint_dir.glob(f"{output_prefix}_*.csv"))
        assert len(all_checkpoints) == 4, f"Erwartet: 4 Checkpoints, gefunden: {len(all_checkpoints)}"
        
        # Überprüfe, ob die Checkpoints in den erwarteten Intervallen erstellt wurden
        for i in range(4):
            expected_checkpoint = checkpoint_dir / f"{output_prefix}_{(i+1)*batch_size}.csv"
            assert expected_checkpoint.exists(), f"Checkpoint für Batch {i+1} fehlt"


def test_resume_checkpoint_naming_pattern(sample_dataframe, mock_client, mock_generate_synthetic_prompt):
    """
    Testet, ob die Checkpoint-Benennung nach einer Wiederaufnahme dem gleichen Muster folgt.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Parameter für den Test
        batch_size = 5
        save_interval = 1
        output_prefix = "resume_test"
        checkpoint_dir = Path(temp_dir)
        
        # Ersten Durchlauf mit 10 Einträgen (2 Batches)
        process_batch(
            df=sample_dataframe,
            batch_size=batch_size,
            start_idx=0,
            save_interval=save_interval,
            fix_errors=False,
            checkpoint_dir=checkpoint_dir,
            output_prefix=output_prefix,
            client=mock_client
        )
        
        # Überprüfe die ersten Checkpoints
        first_run_checkpoints = sorted(list(checkpoint_dir.glob(f"{output_prefix}_*.csv")))
        assert len(first_run_checkpoints) == 4, f"Erwartet: 4 Checkpoints im ersten Durchlauf, gefunden: {len(first_run_checkpoints)}"
        
        # Leere den Inhalt des Verzeichnisses, aber behalte die Struktur
        for file in first_run_checkpoints:
            file.unlink()
        
        # Zweiten Durchlauf ab Index 10 starten mit dem gleichen Präfix
        process_batch(
            df=sample_dataframe,
            batch_size=batch_size,
            start_idx=10,  # Starte von Index 10
            save_interval=save_interval,
            fix_errors=False,
            checkpoint_dir=checkpoint_dir,
            output_prefix=output_prefix,
            client=mock_client
        )
        
        # Überprüfe die Checkpoints vom zweiten Durchlauf
        second_run_checkpoints = sorted(list(checkpoint_dir.glob(f"{output_prefix}_*.csv")))
        assert len(second_run_checkpoints) == 2, f"Erwartet: 2 Checkpoints im zweiten Durchlauf, gefunden: {len(second_run_checkpoints)}"
        
        # Prüfe, ob die Checkpoint-Namen dem erwarteten Muster folgen
        expected_names = [f"{output_prefix}_{(i+1)*batch_size}.csv" for i in range(2, 4)]
        actual_names = [checkpoint.name for checkpoint in second_run_checkpoints]
        
        for expected, actual in zip(expected_names, actual_names):
            assert expected == actual, f"Erwarteter Checkpoint-Name: {expected}, tatsächlicher Name: {actual}"
