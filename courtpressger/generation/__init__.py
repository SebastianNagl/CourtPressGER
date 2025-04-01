"""
Modul zur Generierung von Pressemitteilungen aus Gerichtsurteilen mit
verschiedenen LLMs und synthetischen Prompts.
"""

from .pipeline import LLMGenerationPipeline
from .teuken_generator import generate_with_teuken

__all__ = ["LLMGenerationPipeline", "generate_with_teuken"] 