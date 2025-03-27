"""
Synthetic Prompts Module for CourtPressGER

This module provides functionalities for generating synthetic prompts 
for court rulings and press releases using Claude API.
"""

from .generator import generate_synthetic_prompt, process_batch
from .rate_limiter import RateLimiter, estimate_token_count

__all__ = [
    'generate_synthetic_prompt',
    'process_batch',
    'RateLimiter',
    'estimate_token_count',
]