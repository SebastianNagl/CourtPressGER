"""Process Label Studio annotation results."""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .config import ResultsConfig

logger = logging.getLogger(__name__)


class ResultsProcessor:
    """Processes Label Studio annotation results."""
    
    def __init__(self, config: ResultsConfig):
        """
        Initialize results processor.
        
        Args:
            config: Results processing configuration
        """
        self.config = config
    
    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse ISO timestamp from Label Studio."""
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    
    def _keep_latest_annotations(self, tasks: List[dict]) -> List[dict]:
        """Keep only the latest annotation per case."""
        newest: Dict[str, Tuple[datetime, dict]] = {}
        
        for task in tasks:
            case_id = task.get("case_id") or str(task["id"])
            ts = self._parse_timestamp(task["updated_at"])
            
            if case_id not in newest or ts > newest[case_id][0]:
                newest[case_id] = (ts, task)
        
        logger.info(f"Filtered to {len(newest)} unique cases (latest annotations only)")
        return [pair[1] for pair in newest.values()]
    
    def _validate_annotation(self, task: dict) -> List[str]:
        """Validate a single annotation."""
        missing = []
        
        # Check rank data
        rank_block = task.get("rank")
        if not isinstance(rank_block, list) or not rank_block:
            missing.append("rank")
        else:
            rank_list = rank_block[0].get("ranker", {}).get("rank")
            if not isinstance(rank_list, list) or len(rank_list) != self.config.expected_rank_count:
                missing.append("rank[0].ranker.rank")
        
        # Check quality fields
        for i in range(1, self.config.expected_rank_count + 1):
            for prefix in self.config.quality_prefixes:
                key = f"{prefix}{i}"
                if key in task and task[key] in (None, ""):
                    missing.append(key)
        
        return missing
    
    def _extract_bool_value(self, prefix: str, value: Optional[str]) -> bool:
        """Convert Label Studio annotation to boolean."""
        if value is None:
            return False
        
        text = value.lower()
        
        if prefix == "hallucination":
            return "hallucination" in text or "outside of case" in text
        elif prefix == "incoherent":
            return "incoherent" in text
        elif prefix == "publishable":
            return "publishable" in text
        
        return False
    
    def process_results(self,
                       results_path: str,
                       mapping_path: str,
                       output_dir: str) -> Tuple[Path, Path]:
        """
        Process Label Studio results.
        
        Args:
            results_path: Path to Label Studio results JSON
            mapping_path: Path to press-model mapping JSON
            output_dir: Output directory
            
        Returns:
            Tuple of (processed_results_path, metrics_path)
        """
        logger.info(f"Loading results from {results_path}")
        
        # Load data
        with open(results_path, 'r', encoding='utf-8') as f:
            raw_tasks = json.load(f)
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            press_mapping = json.load(f)
        
        # Filter to latest annotations
        tasks = self._keep_latest_annotations(raw_tasks)
        
        # Validate annotations
        validation_errors = {}
        for task in tasks:
            case_id = task.get("case_id", task.get("id", "unknown"))
            errors = self._validate_annotation(task)
            if errors:
                validation_errors[case_id] = errors
        
        if validation_errors:
            logger.error("Validation errors found:")
            for case_id, errors in validation_errors.items():
                logger.error(f"  {case_id}: {', '.join(errors)}")
            raise ValueError("Invalid annotations found")
        
        logger.info("All annotations validated successfully")
        
        # Build model lookup
        model_lookup = {
            (row["case_id"], row["press_id"]): row["model_name"]
            for row in press_mapping
        }
        
        # Process annotations
        flat_results = []
        
        for task in tasks:
            case_id = task["case_id"]
            rank_list = task["rank"][0].get("ranker", {}).get("rank", [])
            rank_positions = {press_id: idx + 1 for idx, press_id in enumerate(rank_list)}
            
            for press_id, rank in rank_positions.items():
                press_num = press_id[2:]  # "pr7" -> "7"
                
                result = {
                    "case_id": case_id,
                    "press_id": press_id,
                    "model_name": model_lookup.get((case_id, press_id), "UNKNOWN"),
                    "rank": rank
                }
                
                # Add quality assessments
                for prefix in self.config.quality_prefixes:
                    field_key = f"{prefix}{press_num}"
                    result[prefix] = self._extract_bool_value(
                        prefix, 
                        task.get(field_key)
                    )
                
                flat_results.append(result)
        
        # Convert to hierarchical format
        hierarchical_results = self._to_hierarchical(flat_results)
        
        # Compute model metrics
        model_metrics = self._compute_model_metrics(flat_results)
        
        # Save outputs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "processed_results.json"
        metrics_path = output_dir / "model_metrics.json"
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(hierarchical_results, f, ensure_ascii=False, indent=2)
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(model_metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved processed results to {results_path}")
        logger.info(f"Saved model metrics to {metrics_path}")
        
        # Log summary statistics
        logger.info(f"Processed {len(hierarchical_results)} cases")
        logger.info(f"Computed metrics for {len(model_metrics)} models")
        
        return results_path, metrics_path
    
    def _to_hierarchical(self, flat_results: List[dict]) -> List[dict]:
        """Convert flat results to hierarchical format."""
        cases = defaultdict(list)
        
        for result in flat_results:
            case_id = result["case_id"]
            press_info = {k: v for k, v in result.items() if k != "case_id"}
            cases[case_id].append(press_info)
        
        # Sort by rank
        for case_items in cases.values():
            case_items.sort(key=lambda x: x["rank"])
        
        return [
            {"case_id": case_id, "press_items": items}
            for case_id, items in cases.items()
        ]
    
    def _compute_model_metrics(self, flat_results: List[dict]) -> dict:
        """Compute aggregate metrics per model."""
        accumulator = defaultdict(lambda: {
            "count": 0,
            "rank_sum": 0,
            "ranks": [],
            **{f"{prefix}_true": 0 for prefix in self.config.quality_prefixes}
        })
        
        for result in flat_results:
            model = result["model_name"]
            acc = accumulator[model]
            
            acc["count"] += 1
            acc["rank_sum"] += result["rank"]
            acc["ranks"].append(result["rank"])
            
            for prefix in self.config.quality_prefixes:
                if result[prefix]:
                    acc[f"{prefix}_true"] += 1
        
        # Calculate final metrics
        metrics = {}
        
        for model, acc in accumulator.items():
            count = acc["count"]
            metrics[model] = {
                "count": count,
                "avg_rank": round(acc["rank_sum"] / count, 3),
                "ranks": acc["ranks"]
            }
            
            # Add rates for quality metrics
            for prefix in self.config.quality_prefixes:
                rate_key = f"{prefix}_rate"
                metrics[model][rate_key] = round(
                    acc[f"{prefix}_true"] / count, 3
                )
        
        return metrics