"""Label Studio transformation and integration."""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .config import LabelStudioConfig

logger = logging.getLogger(__name__)


class LabelStudioTransformer:
    """Transforms data for Label Studio import."""
    
    def __init__(self, config: LabelStudioConfig):
        """
        Initialize transformer.
        
        Args:
            config: Label Studio configuration
        """
        self.config = config
        self.rng = random.Random(config.random_seed)
    
    def transform_for_label_studio(self, 
                                  input_path: str,
                                  output_dir: str) -> Tuple[Path, Path]:
        """
        Transform augmented data for Label Studio import.
        
        Args:
            input_path: Path to augmented JSON file
            output_dir: Output directory for Label Studio files
            
        Returns:
            Tuple of (tasks_path, mapping_path)
        """
        logger.info(f"Loading augmented data from {input_path}")
        
        # Load input data
        with open(input_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        logger.info(f"Transforming {len(records)} records for Label Studio")
        
        # Prepare output paths
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tasks_path = output_dir / "label_studio_tasks.json"
        mapping_path = output_dir / "press_model_mapping.json"
        
        # Transform records
        ls_tasks = []
        mapping_rows = []
        
        for rec in records:
            case_id = rec["id"]
            prompt = rec["synthetic_prompt"]
            ruling = rec["judgement"]
            court = rec["subset_name"]
            
            # Collect all summaries
            summaries = []
            
            # Add model summaries
            for ms in rec.get("model_summaries", []):
                summaries.append((ms["model_name"], ms["summary"]))
            
            # Add reference summary if configured
            if self.config.include_reference and "summary" in rec:
                summaries.append(("reference_summary", rec["summary"]))
            
            # Shuffle if configured
            if self.config.shuffle_summaries:
                self.rng.shuffle(summaries)
            
            # Create press items
            press_items = []
            press_flat = {}
            
            for idx, (model_name, summary_text) in enumerate(summaries, start=1):
                press_id = f"pr{idx}"
                title = f"PR-{idx}"
                
                press_items.append({
                    "id": press_id,
                    "title": title,
                    "body": summary_text
                })
                
                # Add flat version for Label Studio
                press_flat[f"press{idx}"] = f"{title}\n\n{summary_text}"
                
                # Add to mapping
                mapping_rows.append({
                    "case_id": case_id,
                    "press_id": press_id,
                    "model_name": model_name
                })
            
            # Create Label Studio task
            task = {
                "case_id": case_id,
                "task": f"Rank & evaluate {len(press_items)} AI-generated press releases.",
                "prompt": prompt,
                "court_ruling": ruling,
                "issuing_court": court,
                "press_items": press_items,
                **press_flat  # Add flattened press releases
            }
            
            ls_tasks.append(task)
        
        # Save outputs
        with open(tasks_path, 'w', encoding='utf-8') as f:
            json.dump(ls_tasks, f, ensure_ascii=False, indent=2)
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_rows, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved Label Studio tasks to {tasks_path}")
        logger.info(f"Saved press-model mapping to {mapping_path}")
        
        # Log statistics
        total_summaries = sum(len(task["press_items"]) for task in ls_tasks)
        logger.info(f"Total summaries across all tasks: {total_summaries}")
        
        return tasks_path, mapping_path