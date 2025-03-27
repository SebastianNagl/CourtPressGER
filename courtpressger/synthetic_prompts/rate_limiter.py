"""
Rate limiting functionality for API calls to Anthropic Claude API.

This module provides a RateLimiter class that helps manage the API
rate limits for the Claude API:
- 50 requests per minute
- 20,000 input tokens per minute
- 8,000 output tokens per minute
"""

import time


class RateLimiter:
    """
    Klasse zur Verwaltung der Rate-Limits für die Claude API.
    - 50 Anfragen pro Minute
    - 20.000 Input-Tokens pro Minute
    - 8.000 Output-Tokens pro Minute
    """
    def __init__(self, requests_per_minute=50, input_tokens_per_minute=20000, output_tokens_per_minute=8000):
        self.requests_per_minute = requests_per_minute
        self.input_tokens_per_minute = input_tokens_per_minute
        self.output_tokens_per_minute = output_tokens_per_minute
        
        # Trackers für die letzten 60 Sekunden (Zeitfenster von 1 Minute)
        self.request_timestamps = []
        self.input_token_usage = []  # [timestamp, tokens]
        self.output_token_usage = []  # [timestamp, tokens]
        
    def wait_if_needed(self, input_tokens_estimate, output_tokens_estimate=500):
        """
        Wartet, wenn nötig, um die Rate-Limits einzuhalten.
        Schätzt die Anzahl der benötigten Tokens und wartet, bis genügend Kapazität verfügbar ist.
        """
        current_time = time.time()
        
        # Bereinige abgelaufene Einträge (älter als 60 Sekunden)
        self._clean_expired_entries(current_time)
        
        # Berechne aktuelle Nutzung
        recent_requests = len(self.request_timestamps)
        recent_input_tokens = sum(tokens for _, tokens in self.input_token_usage)
        recent_output_tokens = sum(tokens for _, tokens in self.output_token_usage)
        
        wait_time = 0
        
        # Prüfe Request-Limit
        if recent_requests >= self.requests_per_minute:
            oldest_request = self.request_timestamps[0]
            wait_time = max(wait_time, oldest_request + 60 - current_time)
        
        # Prüfe Input-Token-Limit
        if recent_input_tokens + input_tokens_estimate > self.input_tokens_per_minute:
            wait_time_input = self._calculate_token_wait_time(self.input_token_usage, 
                                                            input_tokens_estimate, 
                                                            self.input_tokens_per_minute)
            wait_time = max(wait_time, wait_time_input)
        
        # Prüfe Output-Token-Limit
        if recent_output_tokens + output_tokens_estimate > self.output_tokens_per_minute:
            wait_time_output = self._calculate_token_wait_time(self.output_token_usage, 
                                                             output_tokens_estimate, 
                                                             self.output_tokens_per_minute)
            wait_time = max(wait_time, wait_time_output)
        
        # Warte, wenn nötig
        if wait_time > 0:
            print(f"Rate-Limit erreicht. Warte {wait_time:.1f} Sekunden...")
            time.sleep(wait_time)
            # Nach dem Warten die abgelaufenen Einträge erneut bereinigen
            current_time = time.time()
            self._clean_expired_entries(current_time)
    
    def _clean_expired_entries(self, current_time):
        """Entfernt Einträge, die älter als 60 Sekunden sind."""
        cutoff_time = current_time - 60
        
        # Bereinige Request-Timestamps
        while self.request_timestamps and self.request_timestamps[0] < cutoff_time:
            self.request_timestamps.pop(0)
        
        # Bereinige Input-Token-Nutzung
        while self.input_token_usage and self.input_token_usage[0][0] < cutoff_time:
            self.input_token_usage.pop(0)
        
        # Bereinige Output-Token-Nutzung
        while self.output_token_usage and self.output_token_usage[0][0] < cutoff_time:
            self.output_token_usage.pop(0)
    
    def _calculate_token_wait_time(self, token_usage, new_tokens, limit):
        """Berechnet die Wartezeit für Token-basierte Limits."""
        if not token_usage:
            return 0
            
        current_usage = sum(tokens for _, tokens in token_usage)
        
        # Wenn das Hinzufügen neuer Tokens das Limit überschreitet, müssen wir warten
        if current_usage + new_tokens > limit:
            # Berechne, wie viele Token wir freigeben müssen
            tokens_to_free = (current_usage + new_tokens) - limit
            
            # Finde den Zeitpunkt, zu dem genügend Token freigegeben werden
            freed_tokens = 0
            for timestamp, tokens in token_usage:
                freed_tokens += tokens
                if freed_tokens >= tokens_to_free:
                    # Warte bis dieser Zeitstempel + 60 Sekunden
                    return timestamp + 60 - time.time()
        
        return 0
    
    def record_usage(self, input_tokens, output_tokens):
        """Zeichnet die Nutzung einer API-Anfrage auf."""
        current_time = time.time()
        self.request_timestamps.append(current_time)
        self.input_token_usage.append([current_time, input_tokens])
        self.output_token_usage.append([current_time, output_tokens])


def estimate_token_count(text):
    """
    Schätzt die Anzahl der Tokens im Text. 
    Einfache Approximation: 1 Token ≈ 4 Zeichen für Englisch/Deutsch.
    """
    return len(text) // 4