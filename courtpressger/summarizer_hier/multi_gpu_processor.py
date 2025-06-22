"""Multi-GPU processing module for parallel execution."""

import os
import math
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd

from .config import ConfigManager

logger = logging.getLogger(__name__)


class MultiGPUProcessor:
    """Handles multi-GPU parallel processing of tasks."""
    
    def __init__(self, config_manager: ConfigManager, task: str, gpu_count: int):
        """
        Initialize multi-GPU processor.
        
        Args:
            config_manager: Configuration manager
            task: Task to perform (chunk, summarize, generate)
            gpu_count: Number of GPUs to use
        """
        self.config_manager = config_manager
        self.task = task
        self.gpu_count = gpu_count
        self.temp_files = []
    
    def split_dataframe(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, str]]:
        """
        Split dataframe into chunks for each GPU.
        
        Args:
            df: Input dataframe
            
        Returns:
            List of (dataframe chunk, temp file path) tuples
        """
        num_rows = len(df)
        chunk_size = math.ceil(num_rows / self.gpu_count)
        
        chunks = []
        for i in range(self.gpu_count):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_rows)
            
            if start_idx >= num_rows:
                break
            
            df_chunk = df.iloc[start_idx:end_idx]
            temp_file = f"temp_{self.task}_gpu_{i}.csv"
            df_chunk.to_csv(temp_file, index=False)
            
            chunks.append((df_chunk, temp_file))
            self.temp_files.append(temp_file)
        
        return chunks
    
    def create_subprocess_command(self, 
                                 gpu_id: int, 
                                 input_file: str, 
                                 output_file: str) -> List[str]:
        """
        Create command for subprocess.
        
        Args:
            gpu_id: GPU ID to use
            input_file: Input file path
            output_file: Output file path
            
        Returns:
            Command list for subprocess
        """
        cmd = [
            "python", "-m", "courtpressger.summarizer_hier.cli",
            self.task,
            "--input", input_file,
            "--output", output_file
        ]
        
        # Add configuration file if available
        if self.config_manager.config_path:
            cmd.extend(["--config", self.config_manager.config_path])
        
        return cmd
    
    def run_parallel_processing(self, input_path: str, output_path: str) -> None:
        """
        Run parallel processing across multiple GPUs.
        
        Args:
            input_path: Input file path
            output_path: Output file path
        """
        logger.info(f"Starting multi-GPU processing with {self.gpu_count} GPUs")
        
        # Load input data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows from {input_path}")
        
        # Split data
        chunks = self.split_dataframe(df)
        logger.info(f"Split data into {len(chunks)} chunks")
        
        # Create output files list
        output_files = []
        processes = []
        
        # Launch processes
        for i, (_, input_file) in enumerate(chunks):
            output_file = f"temp_{self.task}_output_gpu_{i}.csv"
            output_files.append(output_file)
            self.temp_files.append(output_file)
            
            cmd = self.create_subprocess_command(i, input_file, output_file)
            
            # Set GPU environment
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i)
            
            logger.info(f"Launching GPU {i} with command: {' '.join(cmd)}")
            process = subprocess.Popen(cmd, env=env)
            processes.append(process)
        
        # Wait for all processes
        logger.info("Waiting for all processes to complete...")
        for i, process in enumerate(processes):
            process.wait()
            logger.info(f"GPU {i} completed with return code: {process.returncode}")
        
        # Combine results
        logger.info("Combining results...")
        result_dfs = []
        
        for output_file in output_files:
            if os.path.exists(output_file):
                df_result = pd.read_csv(output_file)
                result_dfs.append(df_result)
            else:
                logger.warning(f"Output file not found: {output_file}")
        
        if result_dfs:
            combined_df = pd.concat(result_dfs, ignore_index=True)
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Combined results saved to {output_path}")
        else:
            logger.error("No results to combine")
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self) -> None:
        """Remove temporary files."""
        logger.info("Cleaning up temporary files...")
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Removed {temp_file}")


def run_multi_gpu_processing(task: str, 
                           config_manager: ConfigManager,
                           overrides: dict) -> None:
    """
    Run multi-GPU processing for a given task.
    
    Args:
        task: Task to perform (chunk, summarize, generate)
        config_manager: Configuration manager
        overrides: Configuration overrides
    """
    processing_config = config_manager.get_processing_config(overrides.get('processing'))
    gpu_count = processing_config.gpu_count
    
    if gpu_count <= 1:
        logger.warning("GPU count is 1 or less, use single-GPU processing instead")
        return
    
    processor = MultiGPUProcessor(config_manager, task, gpu_count)
    processor.run_parallel_processing(
        processing_config.input_path,
        processing_config.output_path
    )