# CourtPressGer

## Hintergrund und Ziel
Wir sind Wissenschaftler der Technischen Universität München im Bereich Legal Technology. Für unsere nächste wissenschaftliche Publikation möchten wir drei Dinge vorstellen: 
1. Eine bereinigte Version eines vorab gescrapten Datensatzes mit etwa 6.5k Einträgen von Gerichtsurteilen mit den dazugehörigen Pressemitteilungen und Metadaten.
2. Synthetische Prompts, mit denen man automatisiert aus den Urteilen Pressemitteilungen generieren kann.
3. Eine Evaluation der generierten Pressemitteilungen mit Hilfe von menschlichen und automatisierten Metriken.

## Aktuelle Aufgaben und Probleme



## Funktionalitäten

### Synthetische Prompts

- **Generieren**: Erzeugt synthetische Prompts für Gerichtsurteile und Pressemitteilungen
- **Validieren**: Überprüft CSV-Dateien auf Schema-Konformität
- **Bereinigen**: Korrigiert häufige Probleme in CSV-Dateien
- **Sanitize**: Bereinigt API-Antworten in CSV-Dateien
- **Reparieren**: Behebt strukturelle Probleme in CSV-Dateien

### CSV-Datenmanagement

Das Projekt bietet verschiedene Werkzeuge zur Verwaltung und Qualitätssicherung von CSV-Dateien:

- `validate-csv`: Überprüft CSV-Dateien auf Schema-Konformität
- `clean-csv`: Bereinigt häufige Probleme (doppelte IDs, fehlende Werte, etc.)
- `sanitize-csv`: Bereinigt API-Antworten in CSV-Dateien
- `repair-csv`: Behebt schwerwiegende Strukturprobleme

Beispiele für diese Funktionen finden Sie im `examples/`-Verzeichnis.

## Beispiele

### CSV-Bereinigung

```bash
# Bereinigen einer CSV-Datei
make clean-csv FILE=pfad/zur/datei.csv

# Validieren einer CSV-Datei
make validate-csv FILE=pfad/zur/datei.csv
```

Detaillierte Beispiele und Anleitungen finden Sie in [examples/README.md](examples/README.md).

## Tests

Ausführen aller Tests:

```bash
make test
```

Ausführen spezifischer CSV-Tests:

```bash
make test-csv
```

Dies führt Tests für den CSV-Cleaner und CSV-Validator aus, um sicherzustellen, dass alle Funktionen wie erwartet arbeiten.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz.