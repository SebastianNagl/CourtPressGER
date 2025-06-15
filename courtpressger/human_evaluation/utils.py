"""Utility functions for human evaluation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_csv_columns(csv_path: str, required_columns: List[str]) -> bool:
    """Validate that CSV has required columns."""
    import pandas as pd
    
    df = pd.read_csv(csv_path, nrows=0)  # Just read headers
    missing = set(required_columns) - set(df.columns)
    
    if missing:
        logger.error(f"Missing columns in {csv_path}: {missing}")
        return False
    
    return True


def merge_json_files(file_paths: List[str], output_path: str) -> None:
    """Merge multiple JSON files containing lists."""
    merged = []
    
    for filepath in file_paths:
        data = load_json(filepath)
        if isinstance(data, list):
            merged.extend(data)
        else:
            logger.warning(f"Skipping {filepath}: not a list")
    
    save_json(merged, output_path)
    logger.info(f"Merged {len(file_paths)} files into {output_path}")


def generate_model_report(metrics_path: str, output_path: str) -> None:
    """Generate a human-readable report from model metrics."""
    metrics = load_json(metrics_path)
    
    # Sort models by average rank
    sorted_models = sorted(
        metrics.items(),
        key=lambda x: x[1]["avg_rank"]
    )
    
    # Generate report
    lines = [
        "# Model Evaluation Report",
        "",
        "## Summary",
        f"Total models evaluated: {len(metrics)}",
        "",
        "## Rankings",
        ""
    ]
    
    for rank, (model, data) in enumerate(sorted_models, 1):
        lines.extend([
            f"### {rank}. {model}",
            f"- Average rank: {data['avg_rank']}",
            f"- Total evaluations: {data['count']}",
            f"- Hallucination rate: {data.get('hallucination_rate', 0):.1%}",
            f"- Incoherent rate: {data.get('incoherent_rate', 0):.1%}",
            f"- Publishable rate: {data.get('publishable_rate', 0):.1%}",
            ""
        ])
    
    # Save report
    output_path = Path(output_path)
    output_path.write_text('\n'.join(lines))
    logger.info(f"Generated report: {output_path}")