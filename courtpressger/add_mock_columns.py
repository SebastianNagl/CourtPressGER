import csv
import random
import string
import os

# Configuration
input_file = 'data/processed/cases_prs_synth_prompts_subset.csv'
output_file = 'data/generation/mock_models_results.csv'
models_to_add = ['GPT-4o', 'Llama-3-70B', 'Teuken-7B', 'EuroLLM-9B', 'Llama-3-8B']
min_chars = 300
max_chars = 600

def generate_random_text(length):
    """Generiert einen zufälligen Textblock."""
    # Use German characters and punctuation for slightly more realistic text blocks
    chars = string.ascii_letters + string.digits + ' .,!?äöüÄÖÜß' + ' '*10 # Mehr Leerzeichen für Worttrennung
    return ''.join(random.choice(chars) for i in range(length))

# Process the CSV
try:
    # Zuerst die Header lesen, um Duplikate zu vermeiden
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        try:
            header = next(reader)
        except StopIteration:
            header = []
            print(f"Warnung: Die Eingabedatei '{input_file}' ist leer.")
    
    # Überprüfe, welche Modelle bereits als Spalten existieren und entferne sie aus der Liste
    existing_columns = set(header)
    unique_models = [model for model in models_to_add if model not in existing_columns]
    
    if len(unique_models) < len(models_to_add):
        print(f"Hinweis: {len(models_to_add) - len(unique_models)} Modellspalten existieren bereits und werden nicht dupliziert.")
    
    # Öffne Dateien erneut zum Schreiben
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Header erneut lesen und mit eindeutigen Modellen erweitern
        try:
            header = next(reader)
            new_header = header + unique_models
            writer.writerow(new_header)
        except StopIteration:
            print(f"Fehler: Die Eingabedatei '{input_file}' ist leer oder enthält nur eine Kopfzeile.")
            # Create file with only header if it was empty
            if not header:
                 writer.writerow(unique_models) # Write just new headers if file was totally empty
            # Exit if only header was present but no data rows
            exit()

        # Process rows
        row_count = 0
        for row in reader:
            # Basic check for expected number of columns based on original header
            if len(row) != len(header):
                print(f"Warnung: Zeile {reader.line_num} hat eine unerwartete Anzahl von Spalten ({len(row)} statt {len(header)}). Überspringe die Zeile.")
                continue

            new_data = [generate_random_text(random.randint(min_chars, max_chars)) for _ in unique_models]
            writer.writerow(row + new_data)
            row_count += 1

    # Erfolgsmeldung ausgeben
    if row_count > 0 or (header and row_count == 0): # Ensure we processed rows or had a header
        print(f"Datei '{output_file}' erfolgreich erstellt. {row_count} Datenzeilen verarbeitet.")
        print(f"Hinzugefügte Modellspalten: {', '.join(unique_models)}")
    else:
        print("Keine Datenzeilen zum Verarbeiten gefunden oder geschrieben. Ausgabedatei möglicherweise unvollständig.")
        if os.path.exists(output_file):
            os.remove(output_file)

except FileNotFoundError:
    print(f"Fehler: Eingabedatei '{input_file}' nicht gefunden.")
except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")
    # Clean up temp file if error occurred
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
            print(f"Temporäre Datei '{output_file}' gelöscht.")
        except OSError as rm_e:
            print(f"Fehler beim Löschen der temporären Datei '{output_file}': {rm_e}") 