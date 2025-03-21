# CourtSumGER: German Court Summarization Dataset

This repository contains tools for processing and analyzing the German Court Summarization Dataset (CourtSumGER).

## Setup
1. **Install dependencies from pyproject.toml**: `pip install -e .`
2. **Install optional dependencies**: `pip install -e ".[dev]"`

## Data Sources
- Full dataset: Available at [Huggingface](https://huggingface.co/datasets/rusheeliyer/german-courts)
- Working dataset: After running `cleaning.ipynb`, cleaned data will be available in `cleaned_data/`

## Running the Pipeline
1. Run `cleaning.ipynb` to:
   - Load all available court datasets from Huggingface
   - Apply filtering criteria with detailed statistics
   - Save cleaned datasets to `cleaned_data/`

## Filtering Criteria
The cleaning process applies several filters to ensure quality:
- Valid court case indicators in summary
- Legal structure in judgments
- Semantic relationship between summary and judgment
- No generic press releases or announcements
