import csv
import random
import string
import os

# Configuration
input_file = 'data/processed/cases_prs_synth_prompts_subset.csv'
output_file = 'data/processed/temp_output.csv'
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
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read and write header
        try:
            header = next(reader)
            new_header = header + models_to_add
            writer.writerow(new_header)
        except StopIteration:
            print(f"Fehler: Die Eingabedatei '{input_file}' ist leer oder enthält nur eine Kopfzeile.")
            # Create file with only header if it was empty
            if not header:
                 writer.writerow(models_to_add) # Write just new headers if file was totally empty
            # Exit if only header was present but no data rows
            exit()


        # Process rows
        row_count = 0
        for row in reader:
            # Basic check for expected number of columns based on original header
            if len(row) != len(header):
                print(f"Warnung: Zeile {reader.line_num} hat eine unerwartete Anzahl von Spalten ({len(row)} statt {len(header)}). Überspringe die Zeile.")
                continue

            new_data = [generate_random_text(random.randint(min_chars, max_chars)) for _ in models_to_add]
            writer.writerow(row + new_data)
            row_count += 1

    # Replace original file with the new one only if processing was successful
    if row_count > 0 or (header and row_count == 0): # Ensure we processed rows or had a header
        os.replace(output_file, input_file)
        print(f"Datei '{input_file}' erfolgreich aktualisiert. {row_count} Datenzeilen verarbeitet.")
    else:
        print("Keine Datenzeilen zum Verarbeiten gefunden oder geschrieben. Originaldatei nicht ersetzt.")
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