"""
Synthetische Prompt-Generierung für deutsche Gerichtsurteile und Pressemitteilungen.

Dieses Modul bietet Funktionen für die Generierung synthetischer Zusammenfassungen
von deutschen Gerichtsurteilen unter Verwendung von LLMs.
"""

from .generator import process_batch

__all__ = [
    # Generator
    'process_batch'
]