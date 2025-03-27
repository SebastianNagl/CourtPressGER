"""CourtPress - A package for analyzing court decisions and press releases"""

__version__ = "0.1.0"

from courtpress.models import (
    RuleBasedFilter,
    SemanticSimilarityAnalyzer,
    SupervisedClassifier,
    UnsupervisedClustering,
    CombinedFilter,
    SyntheticPromptGenerator,
)

from courtpress.preprocessing.text_processor import TextProcessor
from courtpress.data.loader import CourtDataLoader
from courtpress.utils.gpu_utils import check_gpu_availability
from courtpress.analysis import DescriptiveAnalyzer

__all__ = [
    # Models
    'RuleBasedFilter',
    'SemanticSimilarityAnalyzer',
    'SupervisedClassifier',
    'UnsupervisedClustering',
    'CombinedFilter',
    'SyntheticPromptGenerator',

    # Preprocessing
    'TextProcessor',

    # Data
    'CourtDataLoader',

    # Utils
    'check_gpu_availability',

    # Analysis
    'DescriptiveAnalyzer',
]
