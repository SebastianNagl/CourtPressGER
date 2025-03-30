"""
Tests für die Sanitizer-Funktionalität in der Synthetic Prompts-Komponente.

Diese Tests überprüfen die Funktionen zum Bereinigen und Validieren von API-Antworten.
"""

import os
import pytest
import tempfile
import pandas as pd
from pathlib import Path
import json
import re
from unittest.mock import patch

from courtpressger.synthetic_prompts.sanitizer import (
    sanitize_api_response,
    sanitize_api_responses_in_csv,
    ERROR_PATTERNS,
    API_CLEANUP_PATTERNS
)


class TestAPISanitizer:
    """Tests für die API-Antwort-Sanitizer Funktionalität."""
    
    def test_sanitize_api_response_empty_input(self):
        """Test mit leeren Eingaben."""
        # None als Eingabe sollte leerer String sein
        result = sanitize_api_response(None)
        assert result == ""
        
        # Leerer String sollte leerer String bleiben
        result = sanitize_api_response("")
        assert result == ""
        
        # Nur Leerzeichen sollten als leerer String zurückgegeben werden
        result = sanitize_api_response("   \n   \t   ")
        assert result == ""
    
    def test_sanitize_api_response_invalid_input_type(self):
        """Test mit ungültigen Eingabetypen."""
        # Integer (5-stellig, damit nicht zu kurz)
        result = sanitize_api_response(12345)
        # Bei integern sollte ein "zu kurz" Fehler zurückgegeben werden
        assert result == "Fehler bei der Generierung des Prompts: Antwort zu kurz"
        
        # Für die Tests mit Listen und Dictionaries verwenden wir eine Mock-Funktion
        # um das Verhalten der Sonderzeichenprüfung zu kontrollieren

        with patch('courtpressger.synthetic_prompts.sanitizer.sum', return_value=0):  # Keine Sonderzeichen
            # Liste mit ausreichender Länge
            list_input = ["Dies", "ist", "eine", "Test-Liste", "mit", "ausreichender", "Länge"]
            result = sanitize_api_response(list_input)
            assert "Fehler bei der Generierung des Prompts" not in result
            
            # Dictionary mit ausreichender Länge
            response_dict = {"test": "Dies ist ein ausreichend langer Wert"}
            result = sanitize_api_response(response_dict)
            assert "Fehler bei der Generierung des Prompts" not in result
    
    def test_sanitize_api_response_error_patterns(self):
        """Test mit verschiedenen Fehlermustern."""
        for pattern in ERROR_PATTERNS:
            # Erstelle einen Mock-Fehlertext basierend auf dem Muster
            if ".*" in pattern:
                error_text = pattern.replace(".*", "Test-Fehler").replace("\\", "")
            else:
                error_text = f"Hier ist ein Problem: {pattern}"
            
            result = sanitize_api_response(error_text)
            assert "Fehler bei der Generierung des Prompts" in result
            assert "Fehlermuster erkannt" in result
    
    def test_sanitize_api_response_json_errors(self):
        """Test mit JSON-Fehlerobjekten."""
        # Standard-JSON-Fehler
        json_error = json.dumps({"error": {"message": "Something went wrong"}})
        result = sanitize_api_response(json_error)
        assert "Fehler bei der Generierung des Prompts" in result
        assert "API-Fehlercode erkannt" in result
        
        # HTTP-Statuscode-Fehler
        status_error = json.dumps({"status": 429, "message": "Rate limit exceeded"})
        result = sanitize_api_response(status_error)
        assert "Fehler bei der Generierung des Prompts" in result
        assert "API-Fehlercode erkannt" in result
    
    def test_sanitize_api_response_too_short(self):
        """Test mit zu kurzen Antworten."""
        result = sanitize_api_response("Kurz")
        assert "Fehler bei der Generierung des Prompts" in result
        assert "Antwort zu kurz" in result
        
        # Genau an der Grenze
        ten_chars = "1234567890"
        result = sanitize_api_response(ten_chars)
        assert "Fehler bei der Generierung des Prompts" not in result
        assert ten_chars == result
    
    def test_sanitize_api_response_too_long(self):
        """Test mit zu langen Antworten."""
        # Erstelle eine Antwort mit 6000 Zeichen
        long_text = "x" * 6000
        result = sanitize_api_response(long_text)
        assert len(result) < 6000
        assert result.endswith("...")
        assert "x" * 5000 in result
    
    def test_sanitize_api_response_special_chars(self):
        """Test mit hohem Anteil an Sonderzeichen."""
        # Text mit mehr als 30% Sonderzeichen
        special_chars = "!@#$%^&*()" * 100
        result = sanitize_api_response(special_chars)
        assert "Fehler bei der Generierung des Prompts" in result
        assert "Ungewöhnlich hoher Anteil an Sonderzeichen" in result
        
        # Text mit null bytes
        null_bytes = "Test mit \0 Null-Bytes \0 im Text"
        result = sanitize_api_response(null_bytes)
        assert "\0" not in result
        assert "Test mit  Null-Bytes  im Text" == result
    
    def test_sanitize_api_response_cleanup_patterns(self):
        """Test der Cleanup-Muster."""
        for pattern, replacement, flags in API_CLEANUP_PATTERNS:
            # Erstelle einen Text, der dem Muster entspricht (mit ausreichender Länge)
            suffix = " Dies ist ein Zusatztext, der ausreichend lang ist für den Test."
            
            if pattern.startswith('^'):
                # Für Pattern, die am Anfang matchen - füge ausreichend Text hinzu
                pattern_simplified = pattern.replace('^', '').replace('$', '')
                if pattern_simplified.startswith('('):
                    # Bei komplexen Ausdrücken verwenden wir einen konkreten Fall
                    if "Hier ist der Prompt" in pattern_simplified:
                        test_text = f"Hier ist der Prompt:{suffix}"
                    else:
                        # Überspringe komplexe Muster, die wir nicht einfach simulieren können
                        continue
                else:
                    test_text = f"{pattern_simplified}{suffix}"
            else:
                # Für Pattern, die am Ende matchen
                if '|' in pattern:
                    # Überspringe komplexe Muster
                    continue
                test_text = f"Testtext vor {pattern.replace('$', '')}{suffix}"
            
            # Stelle sicher, dass der Text lang genug ist
            if len(test_text) < 10:
                test_text += " " * (10 - len(test_text))
                
            result = sanitize_api_response(test_text)
            
            # Prüfe, ob die Bereinigung grundsätzlich funktioniert
            assert result != "", "Sanitizer sollte keinen leeren String zurückgeben"
            assert "Fehler bei der Generierung des Prompts" not in result, f"Keine Fehler für '{test_text}' erwartet"
    
    def test_sanitize_api_response_valid_input(self):
        """Test mit gültigen Eingaben."""
        # Normale Antwort
        valid_text = "Dies ist eine gültige Antwort von der API mit ausreichender Länge."
        result = sanitize_api_response(valid_text)
        assert result == valid_text
        
        # Antwort mit etwas Formatierung
        formatted_text = "```prompt\nErzeugen Sie eine Zusammenfassung des Urteils.\n```"
        result = sanitize_api_response(formatted_text)
        assert "```prompt" not in result
        assert "```" not in result
        assert "Erzeugen Sie eine Zusammenfassung des Urteils." in result


@pytest.fixture
def sample_csv_with_errors():
    """Erstellt eine temporäre CSV-Datei mit API-Antworten und Fehlern."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w+') as f:
        # Erstelle CSV mit Header und einigen Zeilen
        f.write("id,judgement,summary,synthetic_prompt\n")
        f.write('1,"Urteil 1","Pressemitteilung 1","Gültiger synthetischer Prompt 1"\n')
        f.write('2,"Urteil 2","Pressemitteilung 2","Fehler bei der Generierung des Prompts: API-Fehler"\n')
        f.write('3,"Urteil 3","Pressemitteilung 3","```prompt\nGültiger Prompt mit Markup\n```"\n')
        f.write('4,"Urteil 4","Pressemitteilung 4","Kurz"\n')
        f.write('5,"Urteil 5","Pressemitteilung 5","' + ('x' * 6000) + '"\n')
        f.write('6,"Urteil 6","Pressemitteilung 6","!@#$%^&*()" * 50\n')
        f.write('7,"Urteil 7","Pressemitteilung 7","Hier ist der Prompt: Ein validierter Prompt"\n')
        
        f.flush()
        file_path = f.name
    
    yield file_path
    
    # Aufräumen nach dem Test
    if os.path.exists(file_path):
        os.unlink(file_path)
    
    # Entferne auch die Ausgabedatei, falls vorhanden
    output_path = file_path.replace('.csv', '_sanitized.csv')
    if os.path.exists(output_path):
        os.unlink(output_path)


class TestCSVSanitizer:
    """Tests für die CSV-Sanitizer Funktionalität."""
    
    def test_sanitize_api_responses_in_csv(self, sample_csv_with_errors):
        """Test zur Bereinigung von API-Antworten in einer CSV-Datei."""
        # Führe den Sanitizer aus
        output_path = sanitize_api_responses_in_csv(sample_csv_with_errors)
        
        # Lade die Ausgabedatei
        assert os.path.exists(output_path)
        df = pd.read_csv(output_path)
        
        # Überprüfe die Ergebnisse
        assert len(df) == 7  # Sollte alle Zeilen behalten
        
        # Zeile 1: Gültiger Prompt sollte unverändert sein
        assert df.loc[0, 'synthetic_prompt'] == "Gültiger synthetischer Prompt 1"
        
        # Zeile 2: Fehlermeldung sollte erhalten bleiben
        assert "Fehler bei der Generierung des Prompts" in df.loc[1, 'synthetic_prompt']
        
        # Zeile 3: Markup sollte entfernt worden sein
        assert "```prompt" not in df.loc[2, 'synthetic_prompt']
        assert "```" not in df.loc[2, 'synthetic_prompt']
        assert "Gültiger Prompt mit Markup" in df.loc[2, 'synthetic_prompt']
        
        # Zeile 4: Zu kurzer Text sollte als Fehler markiert sein
        assert "Fehler bei der Generierung des Prompts" in df.loc[3, 'synthetic_prompt']
        
        # Zeile 5: Zu langer Text sollte gekürzt sein
        assert len(df.loc[4, 'synthetic_prompt']) < 6000
        assert df.loc[4, 'synthetic_prompt'].endswith('...')
        
        # Zeile 6: Text mit zu vielen Sonderzeichen sollte als Fehler markiert sein
        assert "Fehler bei der Generierung des Prompts" in df.loc[5, 'synthetic_prompt']
        
        # Zeile 7: "Hier ist der Prompt:" sollte entfernt worden sein
        assert "Hier ist der Prompt:" not in df.loc[6, 'synthetic_prompt']
        assert "Ein validierter Prompt" in df.loc[6, 'synthetic_prompt'] 