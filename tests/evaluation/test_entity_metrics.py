"""
Tests für die ContentOverlapMetrics-Klasse, insbesondere die Entity-F1-Metrik.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from courtpressger.evaluation.metrics import ContentOverlapMetrics

# Beispieltexte für Tests
SAMPLE_TEXT_A = """
Das Bundesverwaltungsgericht in Leipzig hat entschieden, dass die Klage der Firma Mueller GmbH gegen das Land Bayern abzuweisen ist. 
Der Vorsitzende Richter Dr. Schmidt begründete die Entscheidung mit Verweis auf § 15 des Verwaltungsverfahrensgesetzes.
"""

SAMPLE_TEXT_B = """
Die Firma Mueller GmbH hat vor dem Bundesverwaltungsgericht in Leipzig gegen das Land Bayern geklagt.
Richter Schmidt wies die Klage unter Berufung auf das Verwaltungsverfahrensgesetz ab.
"""

SAMPLE_TEXT_C = """
Der Verwaltungsgerichtshof hat eine Entscheidung in einem Rechtsstreit getroffen.
Die beteiligten Parteien müssen nun die Konsequenzen tragen.
"""

SAMPLE_TEXT_EMPTY = ""


class MockSpacyModel:
    """Mock für ein spaCy-Modell"""
    
    def __init__(self, entities=None):
        self.entities = entities or []
    
    def __call__(self, text):
        return MagicMock(ents=self.entities)


class MockEntity:
    """Mock für eine spaCy-Entität"""
    
    def __init__(self, text, label):
        self.text = text
        self.label_ = label


# Fixture für das Mocken von SpaCy
@pytest.fixture
def mock_spacy():
    with patch('spacy.load') as mock_load:
        # Mock spaCy-Modell
        mock_load.return_value = MagicMock()
        yield mock_load


@pytest.fixture
def setup_spacy_model_with_entities():
    """Richtet ein Mock-SpaCy-Modell mit vorkonfigurierten Entitäten ein"""
    
    # Entitäten für Text A
    entities_a = [
        MockEntity("Bundesverwaltungsgericht", "ORG"),
        MockEntity("Leipzig", "LOC"),
        MockEntity("Mueller GmbH", "ORG"),
        MockEntity("Land Bayern", "ORG"),
        MockEntity("Dr. Schmidt", "PER")
    ]
    
    # Entitäten für Text B
    entities_b = [
        MockEntity("Mueller GmbH", "ORG"),
        MockEntity("Bundesverwaltungsgericht", "ORG"),
        MockEntity("Leipzig", "LOC"),
        MockEntity("Land Bayern", "ORG"),
        MockEntity("Schmidt", "PER")
    ]
    
    # Entitäten für Text C
    entities_c = [
        MockEntity("Verwaltungsgerichtshof", "ORG")
    ]
    
    # Keine Entitäten für leeren Text
    entities_empty = []
    
    return {
        SAMPLE_TEXT_A: entities_a,
        SAMPLE_TEXT_B: entities_b,
        SAMPLE_TEXT_C: entities_c,
        SAMPLE_TEXT_EMPTY: entities_empty
    }


@pytest.mark.parametrize("reference,generated,expected", [
    # Exakte Übereinstimmung, alle Entitäten
    (SAMPLE_TEXT_A, SAMPLE_TEXT_A, {"entity_precision": 1.0, "entity_recall": 1.0, "entity_f1": 1.0}),
    
    # Teilweise Übereinstimmung
    (SAMPLE_TEXT_A, SAMPLE_TEXT_B, 
     lambda result: 0.0 < result["entity_precision"] < 1.0 and 
                    0.0 < result["entity_recall"] < 1.0 and 
                    0.0 < result["entity_f1"] < 1.0),
    
    # Keine Übereinstimmung
    (SAMPLE_TEXT_A, SAMPLE_TEXT_C, 
     lambda result: result["entity_precision"] == 0.0 or 
                    result["entity_recall"] == 0.0 or 
                    result["entity_f1"] == 0.0),
    
    # Leerer generierter Text
    (SAMPLE_TEXT_A, SAMPLE_TEXT_EMPTY, 
     {"entity_precision": 0.0, "entity_recall": 0.0, "entity_f1": 0.0}),
    
    # Leerer Referenztext
    (SAMPLE_TEXT_EMPTY, SAMPLE_TEXT_A, 
     {"entity_precision": 0.0, "entity_recall": 0.0, "entity_f1": 0.0}),
    
    # Beide Texte leer
    (SAMPLE_TEXT_EMPTY, SAMPLE_TEXT_EMPTY, 
     {"entity_precision": 0.0, "entity_recall": 0.0, "entity_f1": 0.0})
])
def test_entity_f1_real_texts(reference, generated, expected):
    """Test für entity_f1 mit realen Texten (erfordert installierten spaCy)"""
    
    # Überspringe diesen Test, wenn spaCy nicht installiert ist
    pytest.importorskip("spacy")
    
    try:
        # Versuche, das deutsche Sprachmodell zu laden
        import spacy
        spacy.load("de_core_news_sm")
    except OSError:
        pytest.skip("Deutsches SpaCy-Modell 'de_core_news_sm' nicht installiert")
    
    # Führe den Test durch
    result = ContentOverlapMetrics.entity_f1(reference, generated)
    
    # Überprüfe das Ergebnis
    if callable(expected):
        assert expected(result), f"Erwartetes Ergebnis nicht erfüllt: {result}"
    else:
        for key, value in expected.items():
            # Bei NaN-Werten spezielle Prüfung verwenden
            if np.isnan(value):
                assert np.isnan(result[key]), f"Erwartete NaN für {key}, erhielt {result[key]}"
            else:
                assert result[key] == pytest.approx(value), f"Wert für {key} stimmt nicht überein"


@pytest.mark.parametrize("reference_entities,generated_entities,expected", [
    # Übereinstimmende Entitäten
    ([("Berlin", "LOC"), ("Deutschland", "LOC")], [("Berlin", "LOC"), ("Deutschland", "LOC")], 
     {"entity_precision": 1.0, "entity_recall": 1.0, "entity_f1": 1.0}),
    
    # Teilweise Übereinstimmung
    ([("Berlin", "LOC"), ("Deutschland", "LOC")], [("Berlin", "LOC"), ("Paris", "LOC")],
     {"entity_precision": 0.5, "entity_recall": 0.5, "entity_f1": 0.5}),
    
    # Keine Entitäten im generierten Text
    ([("Berlin", "LOC"), ("Deutschland", "LOC")], [],
     {"entity_precision": 1.0, "entity_recall": 0.0, "entity_f1": 0.0}),
    
    # Keine Entitäten im Referenztext
    ([], [("Berlin", "LOC"), ("Deutschland", "LOC")],
     {"entity_precision": 0.0, "entity_recall": 1.0, "entity_f1": 0.0}),
    
    # Keine Entitäten in beiden Texten
    ([], [], {"entity_precision": 1.0, "entity_recall": 1.0, "entity_f1": 1.0})
])
def test_entity_f1_with_mock(mock_spacy, setup_spacy_model_with_entities, reference_entities, generated_entities, expected):
    """Test für entity_f1 mit gemockten Entitäten"""
    
    # Text-Dummy für die Funktion
    ref_text = "Referenztext"
    gen_text = "Generierter Text"
    
    # SpaCy-Mock einrichten
    with patch.object(ContentOverlapMetrics, '_get_nlp') as mock_get_nlp:
        # Mock-Objekt für NLP, das unterschiedliche Entitäten für verschiedene Texte zurückgibt
        mock_nlp = MagicMock()
        mock_nlp.side_effect = lambda text: MagicMock(
            ents=[MockEntity(ent[0], ent[1]) for ent in 
                 (reference_entities if text == ref_text else generated_entities)]
        )
        mock_get_nlp.return_value = mock_nlp
        
        # Führe die Funktion mit Mock-Daten aus
        result = ContentOverlapMetrics.entity_f1(ref_text, gen_text)
        
        # Überprüfe das Ergebnis
        for key, value in expected.items():
            assert result[key] == pytest.approx(value), f"Wert für {key} stimmt nicht überein"


def test_entity_f1_exception_handling(mock_spacy):
    """Test für Fehlerbehandlung in entity_f1"""
    
    # Mock für _get_nlp, der eine Exception auslöst
    with patch.object(ContentOverlapMetrics, '_get_nlp', side_effect=Exception("Test-Fehler")):
        result = ContentOverlapMetrics.entity_f1("Referenztext", "Generierter Text")
        
        # Prüfe, ob NaN-Werte zurückgegeben werden und der Fehler gespeichert wird
        assert np.isnan(result["entity_precision"])
        assert np.isnan(result["entity_recall"])
        assert np.isnan(result["entity_f1"])
        assert "entity_error" in result
        assert "Test-Fehler" in result["entity_error"]


def test_get_nlp_import_error():
    """Test für ImportError-Behandlung in _get_nlp"""
    
    # Temporär SPACY_AVAILABLE auf False setzen
    with patch('courtpressger.evaluation.metrics.SPACY_AVAILABLE', False):
        with pytest.raises(ImportError) as excinfo:
            ContentOverlapMetrics._get_nlp()
        assert "Für entity_f1 wird das Paket 'spacy' benötigt" in str(excinfo.value) 