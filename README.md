# CourtPressGer

## Hintergrund und Ziel
Wir sind Wissenschaftler der Technischen Universität München im Bereich Legal Technology. Für unsere nächste wissenschaftliche Publikation möchten wir drei Dinge vorstellen: 
1. Eine bereinigte Version eines vorab gescrapten Datensatzes mit etwa 6.5k Einträgen von Gerichtsurteilen mit den dazugehörigen Pressemitteilungen und Metadaten. Die Daten können roh über das dataset.py Skript von HF geladen werden. Für die Bereinigung haben wir in courtpressger/data_cleaning eine Pipeline entwickelt.
2. Synthetische Prompts, mit denen man automatisiert aus den Urteilen Pressemitteilungen generieren kann. 
3. Eine Pipeline, die state of the art Modelle mit den synthetischen Pressemitteilungen queryt und die Ergebnisse neben der tatsächlichen Pressemitteilung speichert.
4. Eine Evaluation der generierten Pressemitteilungen mit Hilfe von menschlichen und automatisierten Metriken.
Neben den Pipeline-Skripten haben wir für unsere Analysen auch immer ein Jupyter Notebook für die Arbeitsschritte; abrufbar unter notebooks/.

## Aktuelle Aufgaben und Probleme
Aktuelle Aufgaben können work in progress sein; immer erst mal kontrollieren, dann lösen. Sobald was davon erledigt ist, bitte [erledigt] zu Beginn der Aufgabe schreiben; ich kontrolliere dann bei Gelegenheit. 


# Struktur

## Projektübersicht
Das CourtPressGER-Projekt ist in mehrere Module unterteilt, die verschiedene Funktionalitäten bereitstellen:

1. **Datenbereinigung**: Module zur Reinigung und Vorbereitung der Rohdaten.
2. **Synthetische Prompts**: Module zur Generierung und Validierung synthetischer Prompts für die Erzeugung von Pressemitteilungen aus Gerichtsurteilen.
3. **Generierung**: Module zur Generierung von Pressemitteilungen aus Gerichtsurteilen mit verschiedenen LLMs.
4. **Evaluierung**: Module zur Bewertung der generierten Pressemitteilungen.

## Projektstruktur
```
CourtPressGER/
├── courtpressger/               # Hauptmodul der Anwendung; enthält alle Module und Skripte
│   ├── data_cleaning/           # Module zur Datenbereinigung
│   │   ├── cli.py               # Kommandozeilenschnittstelle für die Datenbereinigung
│   │   ├── rule_based.py        # Regelbasierte Bereinigungsmethoden
│   │   └── utils.py             # Hilfsfunktionen
│   ├── synthetic_prompts/       # Module für synthetische Prompts
│   │   ├── cli.py               # Kommandozeilenschnittstelle für Prompt-Generierung
│   │   ├── generator.py         # Generator für synthetische Prompts
│   │   ├── rate_limiter.py      # API-Ratenbegrenzung
│   │   ├── sanitizer.py         # Bereinigung und Formatierung von Prompts
│   │   └── verify_checkpoints.py # Validierung von Checkpoints
│   ├── generation/              # Module zur Generierung von Pressemitteilungen
│   │   ├── cli.py               # Kommandozeilenschnittstelle für die Generierung
│   │   └── pipeline.py          # Pipeline zur Generierung von Pressemitteilungen
│   ├── evaluation/              # Module zur Evaluierung
│   │   ├── cli.py               # Kommandozeilenschnittstelle für die Evaluierung
│   │   ├── metrics.py           # Metriken für die Evaluierung
│   │   ├── models.py            # Implementierung verschiedener LLMs
│   │   ├── pipeline.py          # Pipeline zur Evaluierung der Modelle
│   │   └── utils.py             # Hilfsfunktionen für Visualisierung und Analyse
│   ├── __init__.py              # Initialisierung des Pakets
│   ├── dataset.py               # Datenverwaltung und -zugriff
│   └── main.py                  # Haupteinstiegspunkt der Anwendung
├── data/                        # Datendateien
│   ├── raw/                     # Rohdaten
│   ├── interim/                 # Zwischendaten
│   ├── processed/               # Verarbeitete Daten
│   ├── checkpoints/             # Checkpoints für lange Verarbeitungsprozesse
│   ├── generation/              # Generierte Pressemitteilungen
│   └── evaluation/              # Evaluierungsergebnisse
├── notebooks/                   # Jupyter Notebooks für Analysen
├── tests/                       # Testmodule
├── models/                      # Modellierte Daten und Modellkonfigurationen
│   ├── eurobert/                # EuroBERT-Modell für BERTScore
│   ├── generation_config.json   # Konfiguration für die Generierungsmodelle
│   └── evaluation_config.json   # Konfiguration für die Evaluierungsmodelle
├── references/                  # Referenzdokumente und Schemadefitionen
├── reports/                     # Generierte Berichte und Visualisierungen
├── pyproject.toml               # Projekteinstellungen und Abhängigkeiten
├── Makefile                     # Automatisierungsscripts für häufige Aufgaben
└── README.md                    # Projektdokumentation
```

## Funktionalitäten
Das Projekt bietet folgende Hauptfunktionalitäten:

### Datenbereinigung
- Bereinigung von Rohdaten mit Gerichtsurteilen und Pressemitteilungen
- Regelbasierte Filterung und Normalisierung
- Semantische Ähnlichkeitsanalyse zur Validierung der Zuordnungen

### Synthetische Prompts
- Generierung von synthetischen Prompts für LLMs
- Verschiedene Prompt-Strategien zur Pressemitteilungsgenerierung
- Funktionen für API-Ratenbegrenzung und Checkpoint-Verwaltung
- Validierung und Bereinigung von generierten Prompts

### Generierung von Pressemitteilungen
- Pipeline zur Generierung von Pressemitteilungen aus Gerichtsurteilen mit verschiedenen LLMs
- Unterstützung für verschiedene Modelltypen:
  - Hugging Face Modelle (german-gpt2, etc.)
  - OpenAI Modelle (GPT-3.5, GPT-4)
  - Lokale Modelle über API-Schnittstellen
- Checkpoint-System zur Fortsetzung unterbrochener Generierungen
- Speicherung der generierten Pressemitteilungen in verschiedenen Formaten (JSON, CSV)

#### Wichtige Unterscheidung: Generierung vs. Evaluierung
In diesem Projekt gibt es zwei separate Pipelines mit unterschiedlichen Aufgaben:
1. **Generierungspipeline**: Verwendet generative Sprachmodelle (z.B. GPT-2, GPT-3.5), um aus Gerichtsurteilen Pressemitteilungen zu erzeugen.
2. **Evaluierungspipeline**: Bewertet die generierten Pressemitteilungen mit verschiedenen Metriken. Hier wird das EuroBERT-Modell nur für die BERTScore-Berechnung verwendet, nicht für die Generierung.

#### Verwendung der Generierungspipeline
```bash
python -m courtpressger.generation.cli \
  --dataset data/processed/cases_prs_synth_prompts_subset.csv \
  --models-config models/generation_config.json \
  --ruling-column judgement \
  --press-column summary \
  --limit 10  # Optional, für Tests
```

#### Modellkonfiguration für die Generierung
Die Konfiguration erfolgt über eine JSON-Datei (models/generation_config.json):
```json
{
    "models": [
        {
            "type": "huggingface",
            "name": "german-gpt2",
            "model_id": "german-nlp-group/german-gpt2",
            "max_length": 512,
            "temperature": 0.7
        },
        {
            "type": "openai",
            "name": "gpt-3.5-turbo",
            "model_name": "gpt-3.5-turbo",
            "max_tokens": 1024,
            "temperature": 0.7
        }
    ]
}
```

### Evaluierung
- Pipeline zur Evaluierung verschiedener LLMs
- Unterstützung für OpenAI, Hugging Face und lokale Modelle
- Berechnung verschiedener Metriken zur Textähnlichkeit:
  - ROUGE (Rouge-1, Rouge-2, Rouge-L)
  - BLEU (BLEU-1 bis BLEU-4)
  - METEOR
  - BERTScore (verwendet EuroBERT-Modell)
- Visualisierung und Reporting-Funktionen für Ergebnisanalyse
- Checkpoint-System für langläufige Evaluierungen

#### Verwendung der Evaluierungspipeline
Nach der Generierung können die Ergebnisse mit der Evaluierungspipeline bewertet werden:
```bash
python -m courtpressger.evaluation.cli \
  --dataset data/generation/all_models_results.csv \
  --models-config models/evaluation_config.json \
  --bert-score-model models/eurobert \
  --metrics rouge bleu meteor bertscore \
  --ruling-column ruling \
  --press-column reference_press \
  --generated-press-column generated_press
```

### CLI-Tools
- Kommandozeilenschnittstellen für verschiedene Aufgaben:
  - Datenbereinigung (`courtpressger-clean`)
  - Prompt-Generierung (`courtpressger-prompt`)
  - Pressemitteilungsgenerierung (`python -m courtpressger.generation.cli`)
  - Modell-Evaluierung (`python -m courtpressger.evaluation.cli`)
  - CSV-Validierung und -Reparatur

## Package Management
Das Projekt nutzt uv, um Pakete und Venv zu verwalten. Im besten Fall sollen pakete durch uv add hinzugefügt werden, nur im Ausnahmefall durch uv pip install.

### Virtuelle Umgebungen
Das Projekt unterstützt zwei Arten von virtuellen Umgebungen:
- CPU-Umgebung: `make venv-cpu` (Standard)
- GPU-Umgebung: `make venv-gpu` (für CUDA-fähige Systeme)

**Wichtig**: Nach der Erstellung muss die gewünschte Umgebung manuell aktiviert werden, bevor weitere Make-Befehle ausgeführt werden können:
```bash
# Für CPU-Umgebung
source .venv-cpu/bin/activate

# Für GPU-Umgebung
source .venv-gpu/bin/activate
```

## Daten
Die Daten liegen im `data` Ordner und in verschiedenen Subordnern. Abhängig vom Bearbeitungsstand gibt es raw, intermediate und processed Daten. Daneben gibt es `checkpoints`, in denen die Zwischenergebnisse der automatisierten Verarbeitung gespeichert werden.

## Lizenz
Dieses Projekt steht unter der MIT-Lizenz.