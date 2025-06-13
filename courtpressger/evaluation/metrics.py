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

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

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
            print(f"INFO: Berechne BERTScore mit lokalem Modell '{self.model_type}' und num_layers=12 (bert-score v0.3.12)...")
            try:
                # BERTScore mit dem angegebenen Modell berechnen 
                P, R, F1 = bert_score.score(
                    [generated], [reference],
                    lang=self.lang,
                    model_type=self.model_type,
                    num_layers=12,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            except Exception as e:
                # Bei Problemen mit EuroBERT, verwenden wir ein direkt unterstütztes Modell
                fallback_model = "bert-base-multilingual-cased"  # Dieses Modell ist in bert-score direkt unterstützt
                print(f"WARNUNG: BERTScore mit {self.model_type} fehlgeschlagen ({e}). Fallback zu {fallback_model}...")
                
                # Fallback mit Standard lang-Parameter, damit die automatischen Layer-Werte verwendet werden
                P, R, F1 = bert_score.score(
                    [generated], [reference],
                    lang=self.lang,
                    model_type=self.model_type
                )
            
            return {
                "bertscore_precision": P.item(),
                "bertscore_recall": R.item(),
                "bertscore_f1": F1.item()
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
    
    # Klassenvariable für SpaCy-Modell, wird bei Bedarf initialisiert
    _nlp = None
    
    @staticmethod
    def _get_nlp():
        """
        Lädt das SpaCy-Modell für Deutsch, wenn es noch nicht geladen wurde.
        
        Returns:
            SpaCy-NLP-Modell
        """
        if not SPACY_AVAILABLE:
            raise ImportError(
                "Für entity_f1 wird das Paket 'spacy' benötigt. "
                "Installiere es mit 'uv add spacy' und lade dann das deutsche Modell mit "
                "'python -m spacy download de_core_news_sm'."
            )
        
        if ContentOverlapMetrics._nlp is None:
            try:
                ContentOverlapMetrics._nlp = spacy.load("de_core_news_sm")
                print("SpaCy-Modell 'de_core_news_sm' erfolgreich geladen.")
            except OSError:
                raise ImportError(
                    "Das deutsche SpaCy-Modell 'de_core_news_sm' ist nicht installiert. "
                    "Installiere es mit 'python -m spacy download de_core_news_sm'."
                )
        
        return ContentOverlapMetrics._nlp
    
    @staticmethod
    def _preprocess_text(text: str) -> List[str]:
        """
        Vorverarbeitung des Textes für die Schlüsselwortextraktion.
        
        Args:
            text: Zu verarbeitender Text
            
        Returns:
            Liste der Wörter nach Vorverarbeitung
        """
        # Zu Kleinbuchstaben umwandeln
        text = text.lower()
        
        # Entferne Sonderzeichen und Zahlen
        text = re.sub(r'[^\w\s]|[\d]', ' ', text)
        
        # Entferne überschüssige Leerzeichen
        text = re.sub(r'\s+', ' ', text).strip()
        
        # In Wörter aufteilen
        return text.split()
    
    @staticmethod
    def keyword_overlap(reference: str, generated: str, num_keywords: int = 20) -> float:
        """
        Berechnet die Überlappung von Schlüsselwörtern zwischen Referenz- und generiertem Text.
        
        Args:
            reference: Referenztext
            generated: Generierter Text
            num_keywords: Anzahl der zu extrahierenden Top-Schlüsselwörter
            
        Returns:
            Jaccard-Ähnlichkeit der Top-Schlüsselwörter (0-1)
        """
        # Texte vorverarbeiten
        ref_words = ContentOverlapMetrics._preprocess_text(reference)
        gen_words = ContentOverlapMetrics._preprocess_text(generated)
        
        # Wörter zählen
        ref_counter = Counter(ref_words)
        gen_counter = Counter(gen_words)
        
        # Stoppwörter herausfiltern (einfache Liste für Deutsch)
        # Eine bessere Lösung wäre, eine vollständige Liste zu verwenden oder nltk/spacy
        stopwords = {"der", "die", "das", "und", "in", "zu", "den", "mit", "von", "für", 
                     "auf", "im", "nicht", "ein", "eine", "ist", "es", "an", "dass", "sich",
                     "bei", "als", "nach", "auch", "vor", "durch", "zum", "zur", "aus", "über"}
        
        for word in stopwords:
            if word in ref_counter:
                del ref_counter[word]
            if word in gen_counter:
                del gen_counter[word]
        
        # Top N-Schlüsselwörter extrahieren
        ref_keywords = set([word for word, _ in ref_counter.most_common(num_keywords)])
        gen_keywords = set([word for word, _ in gen_counter.most_common(num_keywords)])
        
        if not ref_keywords or not gen_keywords:
            return 0.0
        
        # Jaccard-Ähnlichkeit berechnen
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

    @staticmethod
    def entity_f1(reference: str, generated: str) -> Dict[str, float]:
        """
        Berechnet Precision, Recall und F1-Score für benannte Entitäten.
        
        Diese Implementierung verwendet spaCy für die NER-Erkennung.
        
        Args:
            reference: Referenztext
            generated: Generierter Text
            
        Returns:
            Dictionary mit Precision, Recall und F1-Score für Entitäten
        """
        if not reference or not generated:
            return {
                "entity_precision": 0.0,
                "entity_recall": 0.0,
                "entity_f1": 0.0
            }
        
        try:
            nlp = ContentOverlapMetrics._get_nlp()
            
            # Parse Texte mit spaCy
            ref_doc = nlp(reference)
            gen_doc = nlp(generated)
            
            # Extrahiere benannte Entitäten mit ihrer Kategorie
            ref_entities = {(ent.text, ent.label_) for ent in ref_doc.ents}
            gen_entities = {(ent.text, ent.label_) for ent in gen_doc.ents}
            
            # Berechne Precision, Recall und F1
            if not ref_entities and not gen_entities:
                return {
                    "entity_precision": 1.0,  # Wenn keine Entitäten erwartet wurden und keine gefunden wurden
                    "entity_recall": 1.0,
                    "entity_f1": 1.0
                }
            
            if not gen_entities:
                return {
                    "entity_precision": 1.0,  # Wenn keine Entitäten generiert wurden, ist Precision 1
                    "entity_recall": 0.0,     # Aber Recall ist 0, da erwartete Entitäten fehlen
                    "entity_f1": 0.0
                }
            
            if not ref_entities:
                return {
                    "entity_precision": 0.0,  # Wenn keine Entitäten erwartet wurden, aber welche generiert wurden
                    "entity_recall": 1.0,     # Recall ist 1, da es keine erwarteten Entitäten gab
                    "entity_f1": 0.0
                }
            
            # Anzahl der übereinstimmenden Entitäten
            intersection = len(ref_entities.intersection(gen_entities))
            
            # Berechne Metriken
            precision = intersection / len(gen_entities) if gen_entities else 0.0
            recall = intersection / len(ref_entities) if ref_entities else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                "entity_precision": precision,
                "entity_recall": recall,
                "entity_f1": f1
            }
        except Exception as e:
            print(f"Fehler bei der Berechnung des Entity-F1-Scores: {str(e)}")
            return {
                "entity_precision": np.nan,
                "entity_recall": np.nan,
                "entity_f1": np.nan,
                "entity_error": str(e)
            }

class QAGSMetric:
    """
    Implementiert QAGS (Question Answering for evaluating Generated Summaries) Metrik.
    
    QAGS bewertet die sachliche Konsistenz zwischen Quelltexten und generierten Texten.
    Es arbeitet in drei Schritten:
    1. Generierung von Fragen aus dem generierten Text
    2. Beantwortung dieser Fragen mit dem Quelltext (Gerichtsurteil)
    3. Vergleich der Antworten mit denen, die aus dem generierten Text extrahiert wurden
    
    Referenz: Wang et al., 2020, "Asking and Answering Questions to Evaluate the Factual Consistency of Summaries"
    """
    
    def __init__(self, 
                model_name: str = "google/flan-t5-large",
                device: Optional[str] = None,
                max_questions: int = 5,
                lang: str = "de"):
        """
        Initialisiert die QAGS-Metrik.
        
        Args:
            model_name: Name des Modells für Fragen-Generierung und Antwort-Extraktion
            device: Gerät für die Berechnung ('cpu' oder 'cuda')
            max_questions: Maximale Anzahl von Fragen, die generiert werden sollen
            lang: Sprachcode (z.B. 'de' für Deutsch)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Für QAGS werden die Pakete 'transformers' und 'torch' benötigt. "
                "Installiere sie mit 'uv add transformers torch'."
            )
        
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_questions = max_questions
        self.lang = lang
        
        # Lade das Modell für Fragen-Generierung und QA
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            print(f"Lade QAGS QA-Modell: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        except Exception as e:
            print(f"Fehler beim Laden des QAGS-Modells: {e}")
            raise
    
    def generate_questions(self, text: str) -> List[str]:
        """
        Generiert Fragen aus dem übergebenen Text.
        
        Args:
            text: Text, aus dem Fragen generiert werden sollen
            
        Returns:
            Liste von generierten Fragen
        """
        # Prompt für die Fragengenerierung (T5-Format)
        prompt = f"generate questions: {text}"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            
            # Generiere Fragen
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_return_sequences=self.max_questions,
                num_beams=self.max_questions * 2,
                temperature=0.7,
                diversity_penalty=0.5,
                num_beam_groups=self.max_questions
            )
            
            questions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            # Filtere ungültige oder zu kurze Fragen
            valid_questions = [q for q in questions if len(q.split()) >= 3 and ("?" in q)]
            
            return valid_questions[:self.max_questions]
        except Exception as e:
            print(f"Fehler bei der Fragengenerierung: {e}")
            return []
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Beantwortet eine Frage basierend auf dem gegebenen Kontext.
        
        Args:
            question: Die zu beantwortende Frage
            context: Der Kontext, aus dem die Antwort extrahiert werden soll
            
        Returns:
            Die extrahierte Antwort
        """
        # T5-Format für QA
        prompt = f"question: {question} context: {context}"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                temperature=0.7,
            )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return answer.strip()
        except Exception as e:
            print(f"Fehler bei der Antwortextraktion: {e}")
            return ""
    
    def compute(self, source: str, generated: str) -> Dict[str, float]:
        """
        Berechnet die QAGS-Metrik zwischen Quelltext und generiertem Text.
        
        Args:
            source: Quelltext (z.B. Gerichtsurteil)
            generated: Generierter Text (z.B. Pressemitteilung)
            
        Returns:
            Dictionary mit QAGS-Score und zusätzlichen Metriken
        """
        # Generiere Fragen aus dem generierten Text
        questions = self.generate_questions(generated)
        
        if not questions:
            return {
                "qags_score": np.nan,
                "qags_question_count": 0,
                "qags_error": "Keine gültigen Fragen generiert"
            }
        
        # Beantworte jede Frage mit beiden Texten
        source_answers = []
        generated_answers = []
        
        for question in questions:
            source_answer = self.answer_question(question, source)
            generated_answer = self.answer_question(question, generated)
            
            if source_answer and generated_answer:
                source_answers.append(source_answer)
                generated_answers.append(generated_answer)
        
        # Wenn keine Antworten gefunden wurden
        if not source_answers or not generated_answers:
            return {
                "qags_score": np.nan,
                "qags_question_count": len(questions),
                "qags_error": "Keine gültigen Antwortpaare gefunden"
            }
        
        # Berechne Ähnlichkeit der Antworten
        scores = []
        
        try:
            # BERTScore für Antwort-Vergleiche verwenden
            if BERT_SCORE_AVAILABLE:
                P, R, F1 = bert_score.score(
                    generated_answers, 
                    source_answers,
                    lang=self.lang,
                    rescale_with_baseline=True,
                    device=self.device
                )
                
                # F1-Scores in Liste konvertieren
                scores = F1.tolist()
            else:
                # Fallback: Einfache Textüberlappung
                for src_ans, gen_ans in zip(source_answers, generated_answers):
                    src_words = set(src_ans.lower().split())
                    gen_words = set(gen_ans.lower().split())
                    
                    if not src_words or not gen_words:
                        continue
                    
                    overlap = len(src_words.intersection(gen_words)) / len(src_words.union(gen_words))
                    scores.append(overlap)
        except Exception as e:
            print(f"Fehler bei der Antwortvergleichsberechnung: {e}")
            return {
                "qags_score": np.nan,
                "qags_question_count": len(questions),
                "qags_error": f"Fehler beim Antwortvergleich: {str(e)}"
            }
        
        if not scores:
            return {
                "qags_score": np.nan,
                "qags_question_count": len(questions),
                "qags_error": "Keine gültigen Scores berechnet"
            }
        
        # Gesamtergebnis
        return {
            "qags_score": sum(scores) / len(scores),
            "qags_question_count": len(questions),
            "qags_valid_answers": len(scores)
        }

class FactCCMetric:
    """
    Implementiert FactCC für die Bewertung der faktischen Konsistenz.
    
    FactCC ist ein spezialisiertes Modell zur Überprüfung der faktischen Konsistenz
    zwischen einem Quelltext und einem generierten Text. Es wird als binärer Klassifizierer
    trainiert, um zu entscheiden, ob ein generierter Text faktisch konsistent mit dem
    Quelltext ist oder nicht.
    
    Referenz: Kryscinski et al., 2020, "Evaluating the Factual Consistency of Abstractive Text Summarization"
    """
    
    def __init__(self, 
                model_name: str = "google/bert_uncased_L-12_H-768_A-12",
                device: Optional[str] = None,
                lang: str = "de",
                threshold: float = 0.5):
        """
        Initialisiert die FactCC-Metrik.
        
        Args:
            model_name: Name des BERT-basierten Modells für die Konsistenzprüfung
            device: Gerät für die Berechnung ('cpu' oder 'cuda')
            lang: Sprachcode (z.B. 'de' für Deutsch)
            threshold: Schwellenwert für die Klassifizierung (0.0-1.0)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Für FactCC werden die Pakete 'transformers' und 'torch' benötigt. "
                "Installiere sie mit 'uv add transformers torch'."
            )
        
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.lang = lang
        self.threshold = threshold
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            print(f"Lade FactCC-Modell: {model_name}")
            
            # Da es kein vortrainiertes deutsches FactCC gibt, verwenden wir ein BERT-Modell und simulieren FactCC
            # Bei einem echten FactCC-Projekt würde man das Modell auf deutschen Daten feintunen
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=2  # Binäre Klassifikation: konsistent oder inkonsistent
            ).to(self.device)
        except Exception as e:
            print(f"Fehler beim Laden des FactCC-Modells: {e}")
            raise
    
    def extract_claims(self, text: str, max_claims: int = 5) -> List[str]:
        """
        Extrahiert Behauptungen aus dem Text zur Überprüfung.
        
        In einer vollen FactCC-Implementierung würde diese Funktion
        syntaktische Parsing verwenden. Diese Version ist eine Vereinfachung,
        die Sätze als Behauptungen behandelt.
        
        Args:
            text: Text, aus dem Behauptungen extrahiert werden sollen
            max_claims: Maximale Anzahl zu extrahierender Behauptungen
            
        Returns:
            Liste von extrahierten Behauptungen
        """
        # Einfacher Ansatz: Text in Sätze zerlegen
        import re
        
        # Sätze extrahieren (verbesserte Regex für deutsche Texte)
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        
        # Filtere leere Sätze und sehr kurze Sätze heraus
        claims = [s.strip() for s in sentences if len(s.strip().split()) >= 5]
        
        # Beschränke auf max_claims
        return claims[:max_claims]
    
    def check_consistency(self, source: str, claim: str) -> float:
        """
        Überprüft die Konsistenz einer Behauptung mit dem Quelltext.
        
        Args:
            source: Quelltext (z.B. Gerichtsurteil)
            claim: Zu überprüfende Behauptung
            
        Returns:
            Konsistenz-Score zwischen 0.0 und 1.0
        """
        try:
            # Eingabe für das Modell vorbereiten: [CLS] source [SEP] claim [SEP]
            inputs = self.tokenizer(
                source, claim,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            ).to(self.device)
            
            # Vorhersage vom Modell
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Wahrscheinlichkeit für die konsistente Klasse (Index 1 = konsistent, 0 = inkonsistent)
                consistency_score = probs[0, 1].item()
            
            return consistency_score
        except Exception as e:
            print(f"Fehler bei der Konsistenzprüfung: {e}")
            return 0.0
    
    def compute(self, source: str, generated: str) -> Dict[str, float]:
        """
        Berechnet die FactCC-Metrik zwischen Quelltext und generiertem Text.
        
        Args:
            source: Quelltext (z.B. Gerichtsurteil)
            generated: Generierter Text (z.B. Pressemitteilung)
            
        Returns:
            Dictionary mit FactCC-Score und zusätzlichen Metriken
        """
        # Extrahiere Behauptungen aus dem generierten Text
        claims = self.extract_claims(generated)
        
        if not claims:
            return {
                "factcc_score": np.nan,
                "factcc_claim_count": 0,
                "factcc_error": "Keine gültigen Behauptungen extrahiert"
            }
        
        # Überprüfe jede Behauptung
        consistency_scores = []
        
        for claim in claims:
            score = self.check_consistency(source, claim)
            consistency_scores.append(score)
        
        if not consistency_scores:
            return {
                "factcc_score": np.nan,
                "factcc_claim_count": 0,
                "factcc_error": "Keine gültigen Konsistenz-Scores berechnet"
            }
        
        # Berechne durchschnittlichen Konsistenz-Score
        avg_score = sum(consistency_scores) / len(consistency_scores)
        
        # Berechne Anteil der konsistenten Behauptungen
        consistent_claims = sum(1 for score in consistency_scores if score >= self.threshold)
        consistency_ratio = consistent_claims / len(consistency_scores)
        
        return {
            "factcc_score": avg_score,
            "factcc_consistency_ratio": consistency_ratio,
            "factcc_claim_count": len(claims)
        }

class LLMAsJudgeMetric:
    """Verwendet ein LLM (Claude 3.7 Sonnet) als Richter, um Pressemitteilungen zu bewerten."""
    
    def __init__(self, 
                model_name: str = "claude-3-7-sonnet-20250219",
                api_key: Optional[str] = None,
                temperature: float = 0.0,
                max_tokens: int = 1024):
        """
        Initialisiert die LLM-as-a-Judge Metrik.
        
        Args:
            model_name: Anthropic Claude Modell
            api_key: Anthropic API-Schlüssel (wenn nicht angegeben, wird ANTHROPIC_API_KEY verwendet)
            temperature: Temperatur für die Generierung (0 für deterministische Antworten)
            max_tokens: Maximale Anzahl an Tokens für die Ausgabe
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY muss als Parameter oder Umgebungsvariable gesetzt sein")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic-Paket muss installiert sein. Führe 'uv add anthropic' aus.")
    
    def compute(self, source: str, generated: str, reference: Optional[str] = None) -> Dict[str, float]:
        """
        Bewertet die generierte Pressemitteilung im Vergleich zum Quelltext und der Referenz-Pressemitteilung.
        
        Args:
            source: Quelltext (Gerichtsurteil)
            generated: Generierte Pressemitteilung
            reference: Referenz-Pressemitteilung
            
        Returns:
            Dictionary mit den Bewertungsergebnissen
        """
        # Prompt für das Modell vorbereiten
        system_prompt = """
        Du bist ein Experte für juristische Texte und bewertst die Qualität von Pressemitteilungen für Gerichtsurteile.
        Bewerte die generierte Pressemitteilung anhand der folgenden Kriterien auf einer Skala von 1-10:
        
        1. Faktische Korrektheit: Wie genau spiegelt die Pressemitteilung die Fakten aus dem Gerichtsurteil wider?
        2. Vollständigkeit: Wurden alle wichtigen Informationen aus dem Urteil in der Pressemitteilung berücksichtigt?
        3. Klarheit: Wie verständlich ist die Pressemitteilung für ein nicht-juristisches Publikum?
        4. Struktur: Wie gut ist die Pressemitteilung strukturiert und organisiert?
        5. Vergleich mit Referenz: Wie gut ist die generierte Pressemitteilung im Vergleich zur Referenz-Pressemitteilung?
        
        Gib für jedes Kriterium einen numerischen Wert zwischen 1 und 10 an und eine kurze Begründung.
        Berechne abschließend einen Gesamtscore als Durchschnitt aller Einzelwerte.
        Gib deine Antwort im folgenden JSON-Format zurück:
        {
            "faktische_korrektheit": {"wert": X, "begründung": "..."},
            "vollständigkeit": {"wert": X, "begründung": "..."},
            "klarheit": {"wert": X, "begründung": "..."},
            "struktur": {"wert": X, "begründung": "..."},
            "vergleich_mit_referenz": {"wert": X, "begründung": "..."},
            "gesamtscore": X.X
        }
        """
        
        user_prompt = f"""
        # Gerichtsurteil
        {source}
        
        # Generierte Pressemitteilung
        {generated}
        
        # Referenz-Pressemitteilung
        {reference}
        """
        
        try:
            # LLM-Anfrage ausführen
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extrahiere den JSON-Teil aus der Antwort
            response_text = response.content[0].text
            
            # JSON parsen und Ergebnisse zurückgeben
            try:
                import json
                import re
                
                # Extrahiere den JSON-Teil mit einem Regex-Pattern für den Fall, dass es zusätzlichen Text gibt
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                else:
                    result = json.loads(response_text)
                
                # Extrahiere die numerischen Werte für die Metriken
                metrics_result = {}
                
                metrics_result["llm_judge_faktische_korrektheit"] = result["faktische_korrektheit"]["wert"]
                metrics_result["llm_judge_vollständigkeit"] = result["vollständigkeit"]["wert"]
                metrics_result["llm_judge_klarheit"] = result["klarheit"]["wert"]
                metrics_result["llm_judge_struktur"] = result["struktur"]["wert"]
                metrics_result["llm_judge_vergleich_mit_referenz"] = result["vergleich_mit_referenz"]["wert"]
                metrics_result["llm_judge_gesamtscore"] = result["gesamtscore"]
                metrics_result["llm_judge_details"] = response_text
                
                return metrics_result
                
            except Exception as e:
                print(f"Fehler beim Parsen der LLM-Antwort: {e}")
                return {
                    "llm_judge_error": str(e),
                    "llm_judge_raw_response": response_text
                }
                
        except Exception as e:
            print(f"Fehler bei der LLM-Bewertung: {e}")
            return {"llm_judge_error": str(e)}

def compute_all_metrics(reference: str, generated: str, 
                       semantic_similarity_model: Optional[str] = None,
                       bert_score_model: Optional[str] = None,
                       lang: str = "de",
                       source_text: Optional[str] = None,
                       enable_factual_consistency: bool = False,
                       enable_llm_as_judge: bool = False) -> Dict[str, float]:
    """
    Berechnet alle verfügbaren Metriken für ein Paar aus generiertem und Referenztext.
    
    Args:
        reference: Referenztext (tatsächliche Pressemitteilung)
        generated: Generierter Text (generierte Pressemitteilung)
        semantic_similarity_model: Optional, Name des Modells für semantische Ähnlichkeit
        bert_score_model: Optional, Name des Modells für BERTScore
        lang: Sprachcode für BERTScore
        source_text: Optional, Quelltext (Gerichtsurteil) für sachliche Konsistenzmetriken
        enable_factual_consistency: Optional, aktiviert QAGS und FactCC (rechenintensiv)
        enable_llm_as_judge: Optional, aktiviert die LLM-as-a-Judge Bewertung mit Claude Sonnet
        
    Returns:
        Dictionary mit allen berechneten Metriken
    """
    metrics = {}
    
    # # Inhaltliche Überlappungsmetriken
    # metrics['keyword_overlap'] = ContentOverlapMetrics.keyword_overlap(reference, generated)
    # metrics['entity_overlap'] = ContentOverlapMetrics.entity_overlap(reference, generated)
    # metrics['length_ratio'] = ContentOverlapMetrics.length_ratio(reference, generated)
    
    # # Entity-F1-Score berechnen (falls spaCy verfügbar)
    # if SPACY_AVAILABLE:
    #     try:
    #         entity_f1_metrics = ContentOverlapMetrics.entity_f1(reference, generated)
    #         metrics.update(entity_f1_metrics)
    #     except Exception as e:
    #         print(f"Fehler bei der Berechnung des Entity-F1-Scores: {str(e)}")
    
    # # BLEU-Scores berechnen (falls verfügbar)
    # if NLTK_AVAILABLE:
    #     try:
    #         bleu_metrics = BleuMetric.compute(reference, generated)
    #         metrics.update(bleu_metrics)
    #     except Exception as e:
    #         print(f"Fehler bei der Berechnung der BLEU-Scores: {str(e)}")
    
    # # METEOR-Score berechnen (falls verfügbar)
    # if NLTK_AVAILABLE:
    #     try:
    #         metrics['meteor'] = MeteorMetric.compute(reference, generated)
    #     except Exception as e:
    #         print(f"Fehler bei der Berechnung des METEOR-Scores: {str(e)}")
    
    # BERTScore berechnen (falls verfügbar und aktiviert)
    if BERT_SCORE_AVAILABLE and bert_score_model:
        try:
            # Erster Versuch: komplettes Paar auf GPU/CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"INFO: Berechne BERTScore mit Modell '{bert_score_model}' auf {device} und num_layers=12...")
            P, R, F1 = bert_score.score(
                [generated], [reference],
                lang=lang,
                model_type=bert_score_model,
                num_layers=12,
                device=device
            )
        except RuntimeError as e:
            import traceback
            # Prüfen, ob es ein CUDA OOM war
            if "out of memory" in str(e):
                print(f"WARNUNG: CUDA OOM mit {bert_score_model} (num_layers=12). Leere Cache und versuche es mit einem Sample erneut...")
                torch.cuda.empty_cache()
                try:
                    # Zweiter Versuch: nur ein Sample (ist hier identisch, aber bei Batch größerer Daten sinnvoll)
                    P, R, F1 = bert_score.score(
                        [generated], [reference],
                        lang=lang,
                        model_type=bert_score_model,
                        num_layers=12,
                        device=device
                    )
                except RuntimeError as e2:
                    if "out of memory" in str(e2):
                        print(f"WARNUNG: OOM erneut bei Einzel-Sample. Fallback auf kleineres Modell 'bert-base-multilingual-cased' mit Standard-Layern...")
                        torch.cuda.empty_cache()
                        fallback_model = "bert-base-multilingual-cased"
                        # Fallback ohne num_layers, damit bert-score automatisch Layer auswählt
                        P, R, F1 = bert_score.score(
                            [generated], [reference],
                            lang=lang,
                            model_type=fallback_model,
                            device=device
                        )
                    else:
                        # Anderer Fehler beim Einzel-Sample
                        print("Fehler beim BERTScore-Einzel-Sample:", file=sys.stderr)
                        traceback.print_exc()
                        raise
            else:
                # Anderer Fehler beim ersten Lauf
                print(f"Fehler bei der Berechnung des BERTScores mit Modell {bert_score_model}:", file=sys.stderr)
                traceback.print_exc()
                raise
        else:
            # Kein Fehler
            pass

        # Ergebnisse ins metrics-Dict übernehmen
        metrics.update({
            'bertscore_precision': P.item(),
            'bertscore_recall':    R.item(),
            'bertscore_f1':        F1.item()
        })
    
    # # Semantische Ähnlichkeit (falls aktiviert)
    # if semantic_similarity_model and TRANSFORMERS_AVAILABLE:
    #     try:
    #         similarity_metric = SemanticSimilarityMetric(model_name=semantic_similarity_model)
    #         metrics['semantic_similarity'] = similarity_metric.compute(reference, generated)
    #     except Exception as e:
    #         print(f"Fehler bei der Berechnung der semantischen Ähnlichkeit: {str(e)}")
    
    # Sachliche Konsistenzmetriken (QAGS und FactCC, falls aktiviert)
    # if enable_factual_consistency and source_text and TRANSFORMERS_AVAILABLE:
    #     # QAGS
    #     try:
    #         qags_metric = QAGSMetric(lang=lang)
    #         qags_results = qags_metric.compute(source_text, generated)
    #         metrics.update(qags_results)
    #         print(f"INFO: QAGS-Score berechnet: {qags_results.get('qags_score', 'N/A')}")
    #     except Exception as e:
    #         print(f"Fehler bei der Berechnung des QAGS-Scores: {str(e)}")
        
    #     # # FactCC
        # try:
        #     factcc_metric = FactCCMetric(lang=lang)
        #     factcc_results = factcc_metric.compute(source_text, generated)
        #     metrics.update(factcc_results)
        #     print(f"INFO: FactCC-Score berechnet: {factcc_results.get('factcc_score', 'N/A')}")
        # except Exception as e:
        #     print(f"Fehler bei der Berechnung des FactCC-Scores: {str(e)}")
    
    # LLM as a Judge (falls aktiviert)
    # if enable_llm_as_judge and source_text:
    #     try:
    #         llm_judge = LLMAsJudgeMetric()
    #         llm_judge_results = llm_judge.compute(source_text, generated, reference)
    #         metrics.update(llm_judge_results)
    #         print(f"INFO: LLM-as-Judge Score berechnet: {llm_judge_results.get('llm_judge_gesamtscore', 'N/A')}")
    #     except Exception as e:
    #         print(f"Fehler bei der LLM-as-Judge Bewertung: {str(e)}")
    
    return metrics 