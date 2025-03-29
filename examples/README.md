# CSV-Bereinigung mit dem clean-csv Befehl

Diese Beispiele zeigen, wie man den `clean-csv`-Befehl zur Bereinigung von CSV-Dateien verwendet.

## Problematische Daten bereinigen

Die Datei `csv_examples/problematische_daten.csv` enthält verschiedene Probleme:
- Doppelte IDs (ID 1 kommt zweimal vor)
- Fehlende Werte in wichtigen Spalten
- Zeilen mit API-Fehlermeldungen
- Fehlende Werte in der ID-Spalte
- Ungültige Zeichen in Prompts

## Bereinigung durchführen

Führen Sie einen der folgenden Befehle aus, um die Datei zu bereinigen:

```bash
# Mit Python-Modul direkt
python -m courtpressger.synthetic_prompts.cli clean-csv --file examples/csv_examples/problematische_daten.csv

# Mit Makefile (einfacher zu merken)
make clean-csv FILE=examples/csv_examples/problematische_daten.csv
```

## Ausgabe speichern

Um die bereinigte Datei in einer bestimmten Ausgabedatei zu speichern:

```bash
python -m courtpressger.synthetic_prompts.cli clean-csv --file examples/csv_examples/problematische_daten.csv --output examples/csv_examples/bereinigte_daten.csv
```

## Erwartete Ergebnisse

Die Bereinigungsfunktion:
1. Entfernt doppelte IDs (die erste Zeile mit der ID wird behalten)
2. Füllt kleinere Lücken in Daten, wo möglich
3. Entfernt Zeilen mit API-Fehlermeldungen
4. Generiert neue IDs für Zeilen ohne gültige ID
5. Bereinigt ungültige Zeichen in Prompt-Texten

## Workflow für CSV-Datenverwaltung

Typischer Workflow für die Arbeit mit CSV-Dateien:

1. **Validieren**: Prüfen Sie, ob die CSV-Datei dem erwarteten Schema entspricht
   ```bash
   make validate-csv FILE=pfad/zur/datei.csv
   ```

2. **Bereinigen**: Bereinigen Sie die Datei von häufigen Problemen
   ```bash
   make clean-csv FILE=pfad/zur/datei.csv
   ```

3. **Sanitize**: Bereinigen Sie API-Antworten in der CSV-Datei
   ```bash
   make sanitize-csv FILE=pfad/zur/datei.csv
   ```

4. **Reparieren**: Bei schwerwiegenden Strukturproblemen
   ```bash
   make repair-csv FILE=pfad/zur/datei.csv
   ``` 