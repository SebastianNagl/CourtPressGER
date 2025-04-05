# BERTScore-Evaluierung mit EuroBERT

Dieses Dokument beschreibt die Implementierung und Anpassungen, die für die Verwendung von BERTScore-Evaluierungen mit EuroBERT vorgenommen wurden.

## Problemstellung

Bei der Evaluierung von generierten Pressemitteilungen sollte BERTScore mit dem EuroBERT-Modell verwendet werden. Dabei traten folgende Probleme auf:

1. EuroBERT benötigt zwingend den Parameter `trust_remote_code=True`, der nicht automatisch gesetzt wurde.
2. Der Tokenizer von EuroBERT wurde als boolescher Wert zurückgegeben, was zu Fehlern führte, da Methoden wie `sep_token_id` aufgerufen wurden.
3. Die BERTScore-Bibliothek (Version 0.3.12) ist nicht auf das EuroBERT-Modell vorbereitet und führt zu Fehlern.

## Implementierte Lösung

Um die BERTScore-Evaluierung robust zu gestalten, wurden folgende Änderungen vorgenommen:

### 1. Anpassungen in `courtpressger/evaluation/metrics.py`

- Standardmodell für Deutsche Texte von "deepset/gbert-large" zu "bert-base-multilingual-cased" geändert, da dieses in BERTScore direkt unterstützt wird
- Einen Fallback-Mechanismus implementiert, der bei Problemen mit EuroBERT automatisch auf das multilinguale Modell umschaltet
- Parameter `trust_remote_code=True` bei der Initialisierung des BERTScore-Objekts hinzugefügt

```python
# Bei Problemen mit EuroBERT, Fallback verwenden
try:
    P, R, F1 = bert_score.score(
        [generated], [reference],
        lang=lang,
        model_type=bert_score_model,
        num_layers=32,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
except Exception as e:
    # Fallback zu einem direkt unterstützten Modell
    fallback_model = "bert-base-multilingual-cased"  
    print(f"WARNUNG: BERTScore mit {bert_score_model} fehlgeschlagen ({e}). Fallback zu {fallback_model}...")
    
    # Standard lang-Parameter verwenden
    P, R, F1 = bert_score.score(
        [generated], [reference],
        lang=lang,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
```

### 2. Anpassungen in der BERTScore-Bibliothek

Die Python-Bibliothek `bert-score` wurde an zwei Stellen angepasst:

1. In `utils.py`:
   - Die Funktion `get_tokenizer` um `trust_remote_code=True` ergänzt
   - Eine Fehlerbehandlung für boolesche Tokenizer hinzugefügt

2. In `score.py`:
   - Sicherheitsprüfungen hinzugefügt, um zu verhindern, dass auf `sep_token_id` und `cls_token_id` zugegriffen wird, wenn sie nicht existieren

```python
# Anpassungen in score.py
if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
    idf_dict[tokenizer.sep_token_id] = 0
if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
    idf_dict[tokenizer.cls_token_id] = 0
```

## Fallback-Strategie

Als Fallback-Modell wurde "bert-base-multilingual-cased" aus folgenden Gründen gewählt:

1. Es ist in der BERTScore-Bibliothek direkt unterstützt
2. Es handelt sich um ein mehrsprachiges Modell, das Deutsch einschließt
3. Es ist wesentlich kleiner als EuroBERT und lädt daher schneller
4. Es bietet eine gute Balance zwischen Geschwindigkeit und Qualität der Evaluation

## Verwendung

Die BERTScore-Evaluierung kann wie folgt verwendet werden:

```python
# BERTScore mit EuroBERT berechnen (mit automatischem Fallback)
metrics = compute_all_metrics(
    reference=reference_text,
    generated=generated_text,
    bert_score_model="models/eurobert",
    lang="de"
)

# BERTScore-Werte extrahieren
bertscore_precision = metrics.get('bertscore_precision')
bertscore_recall = metrics.get('bertscore_recall')
bertscore_f1 = metrics.get('bertscore_f1')
```

## Hinweise

- Falls EuroBERT korrekt geladen werden kann, wird es für die Berechnung verwendet
- Bei Problemen erfolgt automatisch ein Fallback zum multilingualen Modell
- Es werden entsprechende Warnmeldungen ausgegeben, falls der Fallback aktiviert wird
- Die Metriken werden in allen Fällen berechnet und sind in den Ergebnissen verfügbar 