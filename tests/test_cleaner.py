"""
Tests für die CSV-Cleaner-Funktionen.

Diese Testdatei überprüft die Funktionalität der CSV-Bereinigung und -Reparatur.
"""

import os
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from courtpressger.synthetic_prompts.sanitizer import (
    clean_checkpoint_file,
    verify_csv_file,
    fix_csv_format_errors,
    repair_csv_structure,
    sanitize_api_responses_in_csv,
    sanitize_api_response
)

# Fixture für temporäre Verzeichnisse und Dateien
@pytest.fixture
def temp_dir():
    """Erstellt ein temporäres Verzeichnis für Testdateien."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_clean_checkpoint_file(temp_dir):
    """Test für die clean_checkpoint_file Funktion."""
    # Erstelle eine Testdatei mit API-Fehlermeldungen
    file_path = temp_dir / "error_file.csv"
    content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,Summary 1,Judgement 1,subset1,train,0,criteria1,Fehler bei der Generierung des Prompts: API-Fehler
2,2022-01-02,Summary 2,Judgement 2,subset1,train,0,criteria2,Prompt 2
3,2022-01-03,Summary 3,Judgement 3,subset1,train,0,criteria3,API-Fehler: Rate limit exceeded
4,2022-01-04,Summary 4,Judgement 4,subset1,train,0,criteria4,Prompt 4
5,2022-01-05,Summary 5,Judgement 5,subset1,train,0,criteria5,Your credit balance is too low
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Rufe die Funktion auf
    output_file = clean_checkpoint_file(file_path)
    
    # Überprüfe, ob die Ausgabedatei existiert
    assert os.path.exists(output_file)
    
    # Lade die bereinigte Datei
    df = pd.read_csv(output_file)
    
    # Überprüfe, ob die Zeilen mit Fehlermeldungen entfernt wurden
    assert len(df) == 2  # Nur die Zeilen 2 und 4 sollten übrig bleiben
    assert all(not any(phrase in str(row['synthetic_prompt']) for phrase in [
        "Fehler bei der Generierung des Prompts",
        "API-Fehler",
        "Your credit balance is too low"
    ]) for _, row in df.iterrows())


def test_verify_checkpoint_file(temp_dir):
    """Test für die verify_checkpoint_file Funktion."""
    # Erstelle eine gültige Testdatei
    valid_file_path = temp_dir / "valid_file.csv"
    valid_content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,Summary 1,Judgement 1,subset1,train,0,criteria1,Prompt 1
2,2022-01-02,Summary 2,Judgement 2,subset1,train,0,criteria2,Prompt 2
"""
    with open(valid_file_path, "w", encoding="utf-8") as f:
        f.write(valid_content)
    
    # Erstelle eine ungültige Testdatei
    invalid_file_path = temp_dir / "invalid_file.csv"
    invalid_content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,Summary 1,Judgement 1,subset1,train,0,criteria1,Fehler bei der Generierung des Prompts: API-Fehler
"""
    with open(invalid_file_path, "w", encoding="utf-8") as f:
        f.write(invalid_content)
    
    # Überprüfe die gültige Datei
    is_valid, message = verify_csv_file(valid_file_path)
    assert is_valid
    
    # Überprüfe die ungültige Datei
    is_valid, message = verify_csv_file(invalid_file_path)
    assert not is_valid
    assert "API-Fehlermeldungen" in message


def test_fix_csv_format_errors(temp_dir):
    """Test für die fix_csv_format_errors Funktion."""
    # Erstelle eine Testdatei mit Formatierungsfehlern
    file_path = temp_dir / "format_error_file.csv"
    content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,Summary 1,Judgement 1,subset1,train,0,criteria1,Prompt 1
2,2022-01-02,Summary 2,Judgement 2,subset1,train,0,criteria2,Prompt 2
3,2022-01-03,Summary 3,Judgement 3,subset1,train,0,criteria3
4,2022-01-04,Summary 4,Judgement 4,subset1,train,0,criteria4,Prompt 4,"Extra Column"
5,2022-01-05,Summary 5,Judgement 5,subset1,train,0,criteria5,Prompt 5
6,2022-01-06,Summary 6,Judgement 6,subset1,train,0,criteria6,Prompt 6
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Rufe die Funktion auf
    output_file = fix_csv_format_errors(file_path)
    
    # Überprüfe, ob die Ausgabedatei existiert
    assert os.path.exists(output_file)
    
    # Lade die bereinigte Datei
    df = pd.read_csv(output_file, on_bad_lines='warn')
    
    # Überprüfe, ob nur die gültigen Zeilen übrig bleiben
    assert len(df) > 0  # Es sollten einige Zeilen übrig bleiben
    assert 'id' in df.columns
    assert 'synthetic_prompt' in df.columns


def test_repair_csv_structure(temp_dir):
    """Test für die repair_csv_structure Funktion."""
    # Erstelle eine Testdatei mit beschädigter Struktur
    file_path = temp_dir / "damaged_file.csv"
    content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,Summary 1,Judgement 1,subset1,train,0,criteria1,"This is a multiline
prompt that breaks CSV structure"
2,2022-01-02,Summary 2,Judgement 2,subset1,train,0,criteria2,Prompt 2
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Rufe die Funktion auf
    output_file = repair_csv_structure(file_path)
    
    # Überprüfe, ob die Ausgabedatei existiert
    assert os.path.exists(output_file)
    
    # Lade die reparierte Datei
    df = pd.read_csv(output_file)
    
    # Überprüfe, ob die erwarteten Spalten vorhanden sind
    assert 'id' in df.columns
    assert 'synthetic_prompt' in df.columns


def test_repair_csv_structure_header_only(temp_dir):
    """Test für die repair_csv_structure Funktion mit header_only=True."""
    # Erstelle eine Testdatei mit beschädigter Struktur
    file_path = temp_dir / "damaged_file.csv"
    content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,Summary 1,Judgement 1,subset1,train,0,criteria1,"This is a multiline
prompt that breaks CSV structure"
2,2022-01-02,Summary 2,Judgement 2,subset1,train,0,criteria2,Prompt 2
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Rufe die Funktion mit header_only=True auf
    output_file = repair_csv_structure(file_path, header_only=True)
    
    # Überprüfe, ob die Ausgabedatei existiert
    assert os.path.exists(output_file)
    
    # Lade die reparierte Datei
    df = pd.read_csv(output_file)
    
    # Überprüfe, ob nur der Header vorhanden ist
    assert 'id' in df.columns
    assert 'synthetic_prompt' in df.columns
    assert len(df) == 0  # Keine Daten, nur der Header 


# Test für sanitize_api_response
def test_sanitize_api_response():
    """Test für die sanitize_api_response Funktion."""
    # Test-Fälle mit verschiedenen Eingaben und erwarteten Ausgaben
    test_cases = [
        # Einfacher Prompt ohne Bereinigungsbedarf
        (
            "Fasse das Urteil in einfacher Sprache zusammen und betone die wichtigsten Aspekte.", 
            "Fasse das Urteil in einfacher Sprache zusammen und betone die wichtigsten Aspekte."
        ),
        # Mit Metakommentar
        (
            "Hier ist der Prompt: Fasse das Urteil zusammen und erkläre die rechtlichen Grundlagen.", 
            "Fasse das Urteil zusammen und erkläre die rechtlichen Grundlagen."
        ),
        # Mit Markdown-Formatierung
        (
            "```\nFasse das Urteil in einfacher Sprache zusammen.\n```", 
            "Fasse das Urteil in einfacher Sprache zusammen."
        ),
        # Mit Anführungszeichen
        (
            "\"Fasse das Urteil in prägnanter Form zusammen.\"", 
            "Fasse das Urteil in prägnanter Form zusammen."
        ),
        # Mit Fehlermeldung
        (
            "Fehler bei der Generierung des Prompts: API-Fehler", 
            "Fehler bei der Generierung des Prompts: Fehlermuster erkannt"
        ),
        # Mit anderen Fehlermeldungen
        (
            "Your credit balance is too low for this request", 
            "Fehler bei der Generierung des Prompts: Fehlermuster erkannt"
        ),
        # Mit leerem Text
        (
            "", 
            ""
        ),
        # Mit sehr kurzem Text
        (
            "Kurz", 
            "Fehler bei der Generierung des Prompts: Antwort zu kurz"
        ),
        # Mit Leerzeichen am Anfang und Ende
        (
            "  \n  Fasse das Urteil zusammen.  \n  ", 
            "Fasse das Urteil zusammen."
        ),
    ]
    
    # Führe alle Test-Fälle durch
    for input_text, expected_output in test_cases:
        sanitized = sanitize_api_response(input_text)
        assert sanitized == expected_output, f"Für Eingabe '{input_text}' erwartet: '{expected_output}', erhalten: '{sanitized}'"


# Test für sanitize_api_responses_in_csv
def test_sanitize_api_responses_in_csv(temp_dir):
    """Test für die sanitize_api_responses_in_csv Funktion."""
    # Erstelle eine Testdatei mit unbereinigten API-Antworten
    file_path = temp_dir / "unclean_api_responses.csv"
    content = """id,date,summary,judgement,subset_name,split_name,is_announcement_rule,matching_criteria,synthetic_prompt
1,2022-01-01,Summary 1,Judgement 1,subset1,train,0,criteria1,Hier ist der Prompt: Fasse das Urteil zusammen.
2,2022-01-02,Summary 2,Judgement 2,subset1,train,0,criteria2,"```\nFasse das Urteil in einfacher Sprache zusammen.\n```"
3,2022-01-03,Summary 3,Judgement 3,subset1,train,0,criteria3,"Fehler bei der Generierung des Prompts: API-Fehler"
4,2022-01-04,Summary 4,Judgement 4,subset1,train,0,criteria4,"Kurz"
5,2022-01-05,Summary 5,Judgement 5,subset1,train,0,criteria5,"  \n  Erstelle eine Zusammenfassung des Urteils.  \n  "
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Führe die Bereinigung durch
    result_file = sanitize_api_responses_in_csv(file_path)
    
    # Prüfe, ob die Ergebnisdatei erstellt wurde
    assert os.path.exists(result_file)
    
    # Lade die bereinigte Datei
    df = pd.read_csv(result_file)
    
    # Erwartete bereinigte Werte
    expected_values = [
        "Fasse das Urteil zusammen.",
        "Fasse das Urteil in einfacher Sprache zusammen.",
        "Fehler bei der Generierung des Prompts: Fehlermuster erkannt",
        "Fehler bei der Generierung des Prompts: Antwort zu kurz",
        "Erstelle eine Zusammenfassung des Urteils."
    ]
    
    # Prüfe, ob die Werte wie erwartet bereinigt wurden
    for i, expected in enumerate(expected_values):
        assert df.iloc[i]['synthetic_prompt'] == expected, f"Zeile {i+1}: Erwartet '{expected}', erhalten '{df.iloc[i]['synthetic_prompt']}'" 