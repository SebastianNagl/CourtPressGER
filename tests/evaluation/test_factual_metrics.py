"""
Tests für die QAGSMetric und FactCCMetric.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from courtpressger.evaluation.metrics import QAGSMetric, FactCCMetric

# Beispieltexte für Tests
SAMPLE_SOURCE = """
Das Bundesverwaltungsgericht hat entschieden, dass die Klage gegen den Bescheid des beklagten 
Landes abzuweisen ist. Der Kläger hatte gegen die Feststellung als Grundversorger geklagt. 
Das Verwaltungsgericht hat die Klage abgewiesen und das Bundesverwaltungsgericht hat die 
Entscheidung bestätigt. Nach Ansicht des Gerichts entspricht ein Netzgebiet der allgemeinen 
Versorgung dem Gebiet, für das ein Konzessionsvertrag besteht.
"""

SAMPLE_GENERATED = """
Das Bundesverwaltungsgericht hat die Klage gegen einen Bescheid des beklagten Landes abgewiesen. 
Der Kläger hatte gegen seine Feststellung als Grundversorger geklagt. Nach Ansicht des Gerichts 
entspricht ein Netzgebiet der allgemeinen Versorgung dem Gebiet, für das ein Konzessionsvertrag 
zwischen einem Energieversorgungsunternehmen und der Gemeinde besteht.
"""

SAMPLE_INCONSISTENT = """
Das Bundesverwaltungsgericht hat die Klage des beklagten Landes stattgegeben und die Entscheidung
des Verwaltungsgerichts aufgehoben. Der Kläger hatte gegen die Feststellung als Verbraucher geklagt
und das Gericht hat entschieden, dass ein Versorgungsgebiet nicht dem Konzessionsgebiet entspricht.
"""


class MockModel:
    """Mock-Modell für Tests"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def to(self, device):
        return self
    
    def generate(self, **kwargs):
        # Für QAGSMetric.generate_questions
        if kwargs.get('num_return_sequences', 0) > 1:
            return [
                MagicMock(tolist=lambda: [1, 2, 3]),
                MagicMock(tolist=lambda: [4, 5, 6])
            ]
        # Für QAGSMetric.answer_question
        else:
            return [MagicMock(tolist=lambda: [7, 8, 9])]
    
    def __call__(self, **kwargs):
        return MagicMock(
            last_hidden_state=MagicMock(size=lambda: (1, 1, 768)),
            logits=MagicMock()
        )


class MockTokenizer:
    """Mock-Tokenizer für Tests"""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        mock = MagicMock()
        mock.to = lambda device: mock
        mock['attention_mask'] = MagicMock(
            unsqueeze=lambda dim: MagicMock(
                expand=lambda: MagicMock(
                    float=lambda: MagicMock(
                        sum=lambda dim: MagicMock()
                    )
                )
            )
        )
        return mock
    
    def decode(self, output, skip_special_tokens=True):
        # Für QAGSMetric.generate_questions
        if hasattr(output, '__iter__') and not isinstance(output, (str, bytes)):
            if len(output) > 1:
                return "Wer hat die Klage abgewiesen?"
            else:
                return "Was hat das Bundesverwaltungsgericht entschieden?"
        # Für QAGSMetric.answer_question
        else:
            return "Das Bundesverwaltungsgericht"


@pytest.fixture
def mock_imports():
    """Fixture zum Mocken der benötigten Importe"""
    
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
         patch('transformers.AutoModel.from_pretrained') as mock_model, \
         patch('transformers.AutoModelForSeq2SeqLM.from_pretrained') as mock_seq_model, \
         patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_cls_model, \
         patch('torch.matmul') as mock_matmul, \
         patch('torch.nn.functional.softmax') as mock_softmax, \
         patch('torch.nn.functional.normalize') as mock_normalize:
         
        mock_tokenizer.return_value = MockTokenizer()
        mock_model.return_value = MockModel()
        mock_seq_model.return_value = MockModel()
        mock_cls_model.return_value = MockModel()
        mock_matmul.return_value = MagicMock(item=lambda: 0.85)
        mock_softmax.return_value = MagicMock()
        mock_softmax.return_value.__getitem__.return_value = MagicMock(item=lambda: 0.75)
        mock_normalize.return_value = MagicMock()
        
        yield


@pytest.mark.parametrize("generated_text,expected_score", [
    (SAMPLE_GENERATED, lambda x: 'qags_score' in x and not np.isnan(x['qags_score']) or 'qags_error' in x),
    (SAMPLE_INCONSISTENT, lambda x: 'qags_score' in x and not np.isnan(x['qags_score']) or 'qags_error' in x),
    ("", lambda x: 'qags_error' in x and np.isnan(x['qags_score']))
])
def test_qags_metric(mock_imports, generated_text, expected_score):
    """Test für QAGSMetric"""
    
    # QAGSMetric mit Mocks patchen
    with patch.object(QAGSMetric, 'generate_questions', return_value=["Wer hat die Klage abgewiesen?"]) as mock_gen, \
         patch.object(QAGSMetric, 'answer_question', return_value="Das Bundesverwaltungsgericht") as mock_ans:
    
        # QAGSMetric initialisieren
        qags = QAGSMetric(model_name="test-model")
        
        # Metrik berechnen
        result = qags.compute(SAMPLE_SOURCE, generated_text)
        
        # Überprüfen, ob die Ergebnisse den Erwartungen entsprechen
        assert 'qags_score' in result
        assert expected_score(result)
        
        # Bei gültigen Ergebnissen weitere Checks
        if not np.isnan(result.get('qags_score', np.nan)):
            assert 'qags_question_count' in result
            assert result['qags_question_count'] > 0
            
        # Sicherstellen, dass die Mock-Methoden aufgerufen wurden
        if generated_text:
            mock_gen.assert_called_once()
            mock_ans.assert_called()


@pytest.mark.parametrize("generated_text,expected_score", [
    (SAMPLE_GENERATED, lambda x: not np.isnan(x['factcc_score'])),
    (SAMPLE_INCONSISTENT, lambda x: not np.isnan(x['factcc_score'])),
    ("", lambda x: np.isnan(x['factcc_score']))
])
def test_factcc_metric(mock_imports, generated_text, expected_score):
    """Test für FactCCMetric"""
    
    # FactCCMetric initialisieren
    factcc = FactCCMetric(model_name="test-model")
    
    # Metrik berechnen
    result = factcc.compute(SAMPLE_SOURCE, generated_text)
    
    # Überprüfen, ob die Ergebnisse den Erwartungen entsprechen
    assert 'factcc_score' in result
    assert expected_score(result)
    
    # Bei gültigen Ergebnissen weitere Checks
    if not np.isnan(result.get('factcc_score', np.nan)):
        assert 'factcc_claim_count' in result
        assert 'factcc_consistency_ratio' in result
        assert 0 <= result['factcc_consistency_ratio'] <= 1 