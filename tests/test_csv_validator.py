"""
Tests für die CSV-Validierungsfunktionen.

Diese Testdatei überprüft die Schema-Validierung und Datenbereinigung für CSV-Dateien.
"""

import os
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from courtpressger.synthetic_prompts.sanitizer import (
    validate_csv_schema,
    clean_csv_data
)

# Fixture für temporäre Verzeichnisse und Dateien
@pytest.fixture
def temp_dir():
    """Erstellt ein temporäres Verzeichnis für Testdateien."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_validate_csv_schema_valid(temp_dir):
    """Test für die validate_csv_schema Funktion mit einer gültigen CSV-Datei."""
    # Erstelle eine gültige Testdatei
    file_path = temp_dir / "valid_file.csv"
    content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,Summary 1,Judgement 1,subset1,train,0,criteria1,Prompt 1
2,2022-01-02,Summary 2,Judgement 2,subset1,train,0,criteria2,Prompt 2
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Überprüfe mit der Validierungsfunktion
    is_valid, validation_errors = validate_csv_schema(file_path)
    
    # Die Datei sollte gültig sein
    assert is_valid
    assert not validation_errors


def test_validate_csv_schema_missing_values(temp_dir):
    """Test für die validate_csv_schema Funktion mit fehlenden Werten."""
    # Erstelle eine Testdatei mit fehlenden Werten
    file_path = temp_dir / "missing_values.csv"
    content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,,Judgement 1,subset1,train,0,criteria1,Prompt 1
2,2022-01-02,Summary 2,,subset1,train,0,criteria2,Prompt 2
3,2022-01-03,Summary 3,Judgement 3,,,0,criteria3,Prompt 3
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Überprüfe mit der Validierungsfunktion
    is_valid, validation_errors = validate_csv_schema(file_path)
    
    # Die Datei sollte ungültig sein
    assert not is_valid
    assert validation_errors
    assert any("Fehlende Werte" in str(error) for error in validation_errors)


def test_validate_csv_schema_duplicate_ids(temp_dir):
    """Test für die validate_csv_schema Funktion mit doppelten IDs."""
    # Erstelle eine Testdatei mit doppelten IDs
    file_path = temp_dir / "duplicate_ids.csv"
    content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,Summary 1,Judgement 1,subset1,train,0,criteria1,Prompt 1
1,2022-01-02,Summary 2,Judgement 2,subset1,train,0,criteria2,Prompt 2
3,2022-01-03,Summary 3,Judgement 3,subset1,train,0,criteria3,Prompt 3
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Überprüfe mit der Validierungsfunktion
    is_valid, validation_errors = validate_csv_schema(file_path)
    
    # Die Datei sollte ungültig sein
    assert not is_valid
    assert validation_errors
    assert any("Doppelte IDs" in str(error) for error in validation_errors)


def test_clean_csv_data_duplicates(temp_dir):
    """Test für die clean_csv_data Funktion mit doppelten IDs."""
    # Erstelle eine Testdatei mit doppelten IDs
    file_path = temp_dir / "duplicates_to_clean.csv"
    content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,Summary 1,Judgement 1,subset1,train,0,criteria1,Prompt 1
1,2022-01-02,Summary 2,Judgement 2,subset1,train,0,criteria2,Prompt 2
3,2022-01-03,Summary 3,Judgement 3,subset1,train,0,criteria3,Prompt 3
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Rufe die Bereinigungsfunktion auf
    output_file = clean_csv_data(file_path)
    
    # Überprüfe, ob die Ausgabedatei existiert
    assert os.path.exists(output_file)
    
    # Lade die bereinigte Datei
    df = pd.read_csv(output_file)
    
    # Überprüfe, ob doppelte IDs entfernt wurden
    assert len(df) == 2  # Eine der doppelten Zeilen sollte entfernt worden sein
    assert len(df[df['id'] == 1]) == 1
    assert df['id'].nunique() == df['id'].count()  # Alle IDs sollten eindeutig sein


def test_clean_csv_data_missing_values(temp_dir):
    """Test für die clean_csv_data Funktion mit fehlenden Werten."""
    # Erstelle eine Testdatei mit fehlenden Werten
    file_path = temp_dir / "missing_values_to_clean.csv"
    content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,,Judgement 1,subset1,train,0,criteria1,Prompt 1
2,2022-01-02,Summary 2,,subset1,train,0,criteria2,Prompt 2
3,2022-01-03,Summary 3,Judgement 3,,,0,criteria3,Prompt 3
4,2022-01-04,Summary 4,Judgement 4,subset1,train,0,criteria4,
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Rufe die Bereinigungsfunktion auf
    output_file = clean_csv_data(file_path)
    
    # Überprüfe, ob die Ausgabedatei existiert
    assert os.path.exists(output_file)
    
    # Lade die bereinigte Datei
    df = pd.read_csv(output_file)
    
    # Überprüfe, ob Zeilen mit fehlenden Werten in kritischen Spalten entfernt wurden
    # (id, summary, judgement sollten nicht leer sein)
    assert len(df) < 4  # Einige der problematischen Zeilen sollten entfernt worden sein
    assert not df['id'].isnull().any()
    assert not df['summary'].isnull().any()
    assert not df['judgement'].isnull().any() 