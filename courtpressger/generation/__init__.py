"""
Modul zur Generierung von Pressemitteilungen aus Gerichtsurteilen mit
verschiedenen LLMs und synthetischen Prompts.
Verwendet Langchain für eine vereinfachte Integration verschiedener Modelle.
"""

from .pipeline import LLMGenerationPipeline

__all__ = ["LLMGenerationPipeline"] 