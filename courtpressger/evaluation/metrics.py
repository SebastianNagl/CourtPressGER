"""
Metriken für die Evaluierung von generierten Pressemitteilungen.
"""
from typing import Dict, Any, List, Optional, Union
import numpy as np
from collections import Counter
import re
import os
import sys

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

class SemanticSimilarityMetric:
    """Berechnet die semantische Ähnlichkeit zwischen zwei Texten mit Hilfe von Transformer-Modellen."""
    
    def __init__(self, model_name: str = "stsb-distilbert-base"):
        """
        Initialisiert die Metrik mit einem Transformermodell.
        
        Args:
            model_name: Name des zu verwendenden Modells von Hugging Face
        
        Raises:
            ImportError: Wenn die erforderlichen Pakete nicht installiert sind
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Für die semantische Ähnlichkeitsberechnung werden die Pakete 'torch' und 'transformers' benötigt. "
                "Installiere sie mit 'uv add torch transformers'."
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Gerät für die Berechnung wählen (GPU wenn verfügbar)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
    
    def compute(self, reference: str, generated: str) -> float:
        """
        Berechnet die semantische Ähnlichkeit zwischen zwei Texten.
        
        Args:
            reference: Referenztext (tatsächliche Pressemitteilung)
            generated: Generierter Text (generierte Pressemitteilung)
            
        Returns:
            Cosinus-Ähnlichkeitswert zwischen 0 und 1
        """
        # Tokenisierung und Vorverarbeitung
        inputs = self.tokenizer(
            [reference, generated],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Extrahiere Embeddings vom Modell
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Mean-Pooling der Token-Embeddings
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # Normalisierung der Embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Cosinus-Ähnlichkeit berechnen
            similarity = torch.matmul(
                embeddings[0].unsqueeze(0), 
                embeddings[1].unsqueeze(0).transpose(0, 1)
            ).item()
            
        return similarity

class BleuMetric:
    """Berechnet den BLEU-Score zwischen Referenz- und generiertem Text."""
    
    @staticmethod
    def compute(reference: str, generated: str, 
               weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Berechnet den BLEU-Score für die Textpaare.
        
        Args:
            reference: Referenztext (tatsächliche Pressemitteilung)
            generated: Generierter Text (generierte Pressemitteilung)
            weights: Gewichtungen für die n-gramme (standardmäßig gleichmäßig gewichtet)
            
        Returns:
            Dictionary mit BLEU-Scores für verschiedene n-gramm-Kombinationen
        """
        if not NLTK_AVAILABLE:
            raise ImportError(
                "Für die BLEU-Score-Berechnung wird das Paket 'nltk' benötigt. "
                "Installiere es mit 'uv add nltk' und führe dann "
                "'import nltk; nltk.download(\"punkt\")' aus."
            )
        
        # Tokenisierung mit expliziter Sprachangabe
        try:
            reference_tokens = nltk.word_tokenize(reference.lower(), language='german')
            generated_tokens = nltk.word_tokenize(generated.lower(), language='german')
        except LookupError:
            print("NLTK 'punkt' Tokenizer für Deutsch nicht gefunden. Bitte herunterladen:")
            print("import nltk; nltk.download('punkt')")
            raise
        
        # BLEU-Score berechnen mit verschiedenen n-gram-Gewichtungen
        smoothing = SmoothingFunction().method1
        
        # BLEU-1 bis BLEU-4 berechnen
        bleu_scores = {}
        
        # BLEU-1 (nur Unigramme)
        bleu_scores['bleu1'] = sentence_bleu(
            [reference_tokens], generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing
        )
        
        # BLEU-2 (Unigramme und Bigramme)
        bleu_scores['bleu2'] = sentence_bleu(
            [reference_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing
        )
        
        # BLEU-3 (Uni-, Bi- und Trigramme)
        bleu_scores['bleu3'] = sentence_bleu(
            [reference_tokens], generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing
        )
        
        # BLEU-4 (Standard, gleichmäßig gewichtet über alle 4 n-gramme)
        bleu_scores['bleu4'] = sentence_bleu(
            [reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing
        )
        
        # Spezielle Gewichtung, falls angegeben
        if weights:
            bleu_scores['bleu_custom'] = sentence_bleu(
                [reference_tokens], generated_tokens, weights=weights, smoothing_function=smoothing
            )
        
        return bleu_scores

class MeteorMetric:
    """Berechnet den METEOR-Score zwischen Referenz- und generiertem Text."""
    
    @staticmethod
    def compute(reference: str, generated: str) -> float:
        """
        Berechnet den METEOR-Score für die Textpaare.
        
        Args:
            reference: Referenztext (tatsächliche Pressemitteilung)
            generated: Generierter Text (generierte Pressemitteilung)
            
        Returns:
            METEOR-Score
        """
        if not NLTK_AVAILABLE:
            raise ImportError(
                "Für die METEOR-Score-Berechnung wird das Paket 'nltk' benötigt. "
                "Installiere es mit 'uv add nltk' und führe dann "
                "'import nltk; nltk.download(\"wordnet\"); nltk.download(\"omw-1.4\")' aus."
            )
        
        # Tokenisierung mit expliziter Sprachangabe
        try:
            reference_tokens = nltk.word_tokenize(reference.lower(), language='german')
            generated_tokens = nltk.word_tokenize(generated.lower(), language='german')
        except LookupError:
            print("NLTK 'punkt' Tokenizer für Deutsch nicht gefunden. Bitte herunterladen:")
            print("import nltk; nltk.download('punkt')")
            raise
        
        # METEOR-Score berechnen
        return meteor_score([reference_tokens], generated_tokens)

class BertScoreMetric:
    """Berechnet den BERTScore zwischen Referenz- und generiertem Text."""
    
    def __init__(self, model_type: Optional[str] = None, 
                 lang: str = "de", device: Optional[str] = None):
        """
        Initialisiert die BERTScore-Metrik.
        
        Args:
            model_type: Name des zu verwendenden Modells von Hugging Face (z.B. "bert-base-german-cased").
                        Wenn None, wird versucht, ein geeignetes Standardmodell für die Sprache zu verwenden.
            lang: Sprachcode (z.B. 'de' für Deutsch)
            device: Gerät für die Berechnung ('cpu' oder 'cuda')
            
        Raises:
            ImportError: Wenn die erforderlichen Pakete nicht installiert sind
            ValueError: Wenn für die Sprache kein Standardmodell bekannt ist und model_type=None.
        """
        if not BERT_SCORE_AVAILABLE:
            raise ImportError(
                "Für die BERTScore-Berechnung wird das Paket 'bert_score' benötigt. "
                "Installiere es mit 'uv add bert-score torch torchvision torchaudio'."
            )
        if not TRANSFORMERS_AVAILABLE:
             # Bert-score benötigt auch transformers
             raise ImportError(
                "Für die BERTScore-Berechnung wird das Paket 'transformers' benötigt. "
                "Installiere es mit 'uv add transformers'."
             )
        
        self.model_type = model_type
        self.lang = lang
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setze Standardmodell basierend auf der Sprache, wenn keins angegeben ist.
        # Hinweis: EuroBERT ist sehr groß. Standardmäßig verwenden wir hier ggf. ein kleineres.
        # Für spezifische Modelle wie EuroBERT sollte model_type explizit gesetzt werden.
        if self.model_type is None:
            if self.lang == "de":
                # Ein zuverlässiges multilinguales Modell, das in bert-score direkt unterstützt wird
                self.model_type = "bert-base-multilingual-cased"
                print(f"Warnung: Kein BERTScore-Modelltyp angegeben. Verwende Standard für lang='{self.lang}': {self.model_type}")
            # Füge hier ggf. weitere Standardmodelle für andere Sprachen hinzu
            # elif self.lang == "en":
            #     self.model_type = "roberta-large"
            else:
                raise ValueError(f"Kein Standard-BERTScore-Modell für Sprache '{self.lang}' definiert. Bitte model_type angeben.")

    def compute(self, reference: str, generated: str) -> Dict[str, float]:
        """
        Berechnet den BERTScore.
        
        Args:
            reference: Referenztext
            generated: Generierter Text
            
        Returns:
            Dictionary mit Precision, Recall und F1-Score des BERTScores.
        """
        try:
            P, R, F1 = bert_score.score(
                [generated], 
                [reference], 
                lang=self.lang, 
                model_type=self.model_type, 
                device=self.device,
                verbose=False, # Weniger Output in der Konsole
                trust_remote_code=True # Wichtig für EuroBERT
            )
            
            return {
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
                "bertscore_f1": F1.mean().item()
            }
        except Exception as e:
            print(f"Fehler bei BERTScore-Berechnung (Modell: {self.model_type}, Lang: {self.lang}): {e}")
            # Gib NaN oder einen anderen Fehlerindikator zurück
            return {
                "bertscore_precision": np.nan,
                "bertscore_recall": np.nan,
                "bertscore_f1": np.nan,
                "bertscore_error": str(e)
            }

class ContentOverlapMetrics:
    """Berechnet Metriken zur Überlappung von Inhalt zwischen generierten und Referenztexten."""
    
    @staticmethod
    def _preprocess_text(text: str) -> List[str]:
        """
        Vorverarbeitung des Textes für die Berechnung von Überlappungsmetriken.
        
        Args:
            text: Der zu verarbeitende Text
            
        Returns:
            Liste der Wörter im Text (ohne Satzzeichen, Stoppwörter, etc.)
        """
        # Zu Kleinbuchstaben konvertieren
        text = text.lower()
        
        # Satzzeichen entfernen
        text = re.sub(r'[^\w\s]', '', text)
        
        # Aufteilen in Wörter
        words = text.split()
        
        return words
    
    @staticmethod
    def keyword_overlap(reference: str, generated: str, num_keywords: int = 20) -> float:
        """
        Berechnet die Überlappung der häufigsten Schlüsselwörter.
        
        Args:
            reference: Referenztext
            generated: Generierter Text
            num_keywords: Anzahl der zu betrachtenden Schlüsselwörter
            
        Returns:
            Jaccard-Ähnlichkeit der Schlüsselwörter (0-1)
        """
        ref_words = ContentOverlapMetrics._preprocess_text(reference)
        gen_words = ContentOverlapMetrics._preprocess_text(generated)
        
        # Häufigste Wörter in beiden Texten ermitteln
        ref_counter = Counter(ref_words)
        gen_counter = Counter(gen_words)
        
        # Top-N Schlüsselwörter extrahieren
        ref_keywords = set([word for word, _ in ref_counter.most_common(num_keywords)])
        gen_keywords = set([word for word, _ in gen_counter.most_common(num_keywords)])
        
        # Jaccard-Ähnlichkeit berechnen
        if not ref_keywords or not gen_keywords:
            return 0.0
        
        intersection = len(ref_keywords.intersection(gen_keywords))
        union = len(ref_keywords.union(gen_keywords))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def entity_overlap(reference: str, generated: str) -> float:
        """
        Berechnet die Überlappung von benannten Entitäten.
        
        Dies ist eine vereinfachte Implementation, die nach Großbuchstaben sucht.
        Für eine genauere Erkennung sollte ein NER-Modell verwendet werden.
        
        Args:
            reference: Referenztext
            generated: Generierter Text
            
        Returns:
            Jaccard-Ähnlichkeit der Entitäten (0-1)
        """
        # Einfache Heuristik: Wörter, die mit Großbuchstaben beginnen
        ref_entities = set(re.findall(r'\b[A-Z][a-zA-Z]*\b', reference))
        gen_entities = set(re.findall(r'\b[A-Z][a-zA-Z]*\b', generated))
        
        if not ref_entities or not gen_entities:
            return 0.0
        
        intersection = len(ref_entities.intersection(gen_entities))
        union = len(ref_entities.union(gen_entities))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def length_ratio(reference: str, generated: str) -> float:
        """
        Berechnet das Verhältnis der Längen von generiertem und Referenztext.
        
        Args:
            reference: Referenztext
            generated: Generierter Text
            
        Returns:
            Verhältnis der Längen, wobei 1.0 bedeutet, dass beide Texte gleich lang sind
        """
        ref_length = len(reference.split())
        gen_length = len(generated.split())
        
        if ref_length == 0 or gen_length == 0:
            return 0.0
        
        # Skalierung, damit 1.0 eine perfekte Übereinstimmung bedeutet
        ratio = min(ref_length, gen_length) / max(ref_length, gen_length)
        
        return ratio

def compute_all_metrics(reference: str, generated: str, 
                       semantic_similarity_model: Optional[str] = None,
                       bert_score_model: Optional[str] = None,
                       lang: str = "de") -> Dict[str, float]:
    """
    Berechnet alle verfügbaren Metriken für ein Paar aus generiertem und Referenztext.
    
    Args:
        reference: Referenztext (tatsächliche Pressemitteilung)
        generated: Generierter Text (generierte Pressemitteilung)
        semantic_similarity_model: Optional, Name des Modells für semantische Ähnlichkeit
        bert_score_model: Optional, Name des Modells für BERTScore
        lang: Sprachcode für BERTScore
        
    Returns:
        Dictionary mit allen berechneten Metriken
    """
    metrics = {}
    
    # Inhaltliche Überlappungsmetriken
    metrics['keyword_overlap'] = ContentOverlapMetrics.keyword_overlap(reference, generated)
    metrics['entity_overlap'] = ContentOverlapMetrics.entity_overlap(reference, generated)
    metrics['length_ratio'] = ContentOverlapMetrics.length_ratio(reference, generated)
    
    # BLEU-Scores berechnen (falls verfügbar)
    if NLTK_AVAILABLE:
        try:
            bleu_metrics = BleuMetric.compute(reference, generated)
            metrics.update(bleu_metrics)
        except Exception as e:
            print(f"Fehler bei der Berechnung der BLEU-Scores: {str(e)}")
    
    # METEOR-Score berechnen (falls verfügbar)
    if NLTK_AVAILABLE:
        try:
            metrics['meteor'] = MeteorMetric.compute(reference, generated)
        except Exception as e:
            print(f"Fehler bei der Berechnung des METEOR-Scores: {str(e)}")
    
    # BERTScore berechnen (falls verfügbar und aktiviert)
    if BERT_SCORE_AVAILABLE and bert_score_model:
        try:
            # Verwende den ursprünglichen lokalen Pfad
            print(f"INFO: Berechne BERTScore mit lokalem Modell '{bert_score_model}' und num_layers=32 (bert-score v0.3.12)...")

            # Versuchen wir es mit EuroBERT
            try:
                P, R, F1 = bert_score.score(
                    [generated], [reference],
                    lang=lang,
                    model_type=bert_score_model,
                    num_layers=32,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            except Exception as e:
                # Bei Problemen mit EuroBERT, verwenden wir ein direkt unterstütztes Modell
                fallback_model = "bert-base-multilingual-cased"  # Dieses Modell ist in bert-score direkt unterstützt
                print(f"WARNUNG: BERTScore mit {bert_score_model} fehlgeschlagen ({e}). Fallback zu {fallback_model}...")
                
                # Fallback mit Standard lang-Parameter, damit die automatischen Layer-Werte verwendet werden
                P, R, F1 = bert_score.score(
                    [generated], [reference],
                    lang=lang,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            
            metrics.update({
                'bertscore_precision': P.item(),
                'bertscore_recall': R.item(),
                'bertscore_f1': F1.item()
            })
        except Exception as e:
            import traceback
            print(f"Fehler bei der Berechnung des BERTScores mit Modell {bert_score_model}:", file=sys.stderr)
            traceback.print_exc()
    
    # Semantische Ähnlichkeit (falls aktiviert)
    if semantic_similarity_model and TRANSFORMERS_AVAILABLE:
        try:
            similarity_metric = SemanticSimilarityMetric(model_name=semantic_similarity_model)
            metrics['semantic_similarity'] = similarity_metric.compute(reference, generated)
        except Exception as e:
            print(f"Fehler bei der Berechnung der semantischen Ähnlichkeit: {str(e)}")
    
    return metrics 