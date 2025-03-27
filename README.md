# CourtPress

A Python package for analyzing court decisions and press releases, with tools for data preprocessing, descriptive analysis, and synthetic prompt generation.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CourtPressGER.git
cd CourtPressGER

# Install the package
pip install -e .
```

After installation, the following command-line tools will be available:
- `courtpress-clean` - Data cleaning tool
- `courtpress-analyze` - Descriptive analysis tool
- `courtpress-prompts` - Synthetic prompt generation tool
- `courtpress-combined` - Combined analysis tool

## Troubleshooting

If the command-line tools are not available after installation, try the following:

1. Make sure the package is installed in development mode:
   ```bash
   pip install -e .
   ```

2. If you're still experiencing issues, check that entry points are properly configured in the `pyproject.toml` file:
   ```toml
   [project.scripts]
   courtpress-analyze = "courtpress.scripts.run_analysis:main"
   courtpress-clean = "courtpress.scripts.run_cleaning:main"
   courtpress-prompts = "courtpress.scripts.generate_prompts:main"
   courtpress-combined = "courtpress.scripts.run_combined_analysis:main"
   ```

3. You can also run the scripts directly using Python module syntax:
   ```bash
   python -m courtpress.scripts.run_cleaning --help
   python -m courtpress.scripts.run_analysis --help
   python -m courtpress.scripts.generate_prompts --help
   python -m courtpress.scripts.run_combined_analysis --help
   ```

## Features

- **Data Loading**: Load court decisions and press releases from CSV files
- **Data Preprocessing**: Clean and prepare text data for analysis
- **Rule-Based Filtering**: Filter text using rule-based approaches
- **Semantic Similarity Analysis**: Analyze the semantic similarity between court decisions and press releases
- **Supervised Classification**: Classify text using supervised machine learning
- **Unsupervised Clustering**: Cluster text using unsupervised learning
- **Combined Methods**: Apply multiple filtering methods together
- **Descriptive Analysis**: Generate statistical analyses and visualizations
- **Synthetic Prompt Generation**: Create prompts for generating press releases from court decisions

## Data Directory Structure

```
data/
├── raw/          # Raw, unprocessed data
├── processed/    # Cleaned and processed data
└── interim/      # Intermediate data files
```

- Place your original court decision datasets in `data/raw/`
- Cleaned data will be stored in `data/processed/`
- Intermediate analysis files will be stored in `data/interim/`

## Usage

### Using Command-line Tools

#### Running Data Cleaning

```bash
# If installed as a package:
courtpress-clean --methods all --output-dir data/processed

# Or run directly:
python -m courtpress.scripts.run_cleaning --methods all --output-dir data/processed
```

Options:
- `--methods`: Cleaning methods to use (rule, semantic, supervised, unsupervised, combined, all)
- `--output-dir`: Directory to save cleaned data

#### Running Descriptive Analysis

```bash
# If installed as a package:
courtpress-analyze --output-dir analysis_results --save-figures

# Or run directly:
python -m courtpress.scripts.run_analysis --output-dir analysis_results --save-figures
```

Options:
- `--output-dir`: Directory to save analysis results
- `--data-file`: Path to the data file
- `--save-figures`: Save generated figures to output directory

#### Generating Synthetic Prompts

```bash
# If installed as a package:
courtpress-prompts --api-key YOUR_ANTHROPIC_API_KEY --model claude-3-haiku-20240307

# Or run directly:
python -m courtpress.scripts.generate_prompts --api-key YOUR_ANTHROPIC_API_KEY --model claude-3-haiku-20240307
```

Options:
- `--api-key`: Anthropic API key (defaults to ANTHROPIC_API_KEY env variable)
- `--model`: Claude model to use
- `--batch-size`: Batch size
- `--start-idx`: Start index for processing
- `--save-interval`: Save checkpoint interval
- `--sample-size`: Number of samples to use

#### Combined Analysis

```bash
# If installed as a package:
courtpress-combined --descriptive --synthetic --api-key YOUR_ANTHROPIC_API_KEY --save-figures

# Or run directly:
python -m courtpress.scripts.run_combined_analysis --descriptive --synthetic --api-key YOUR_ANTHROPIC_API_KEY --save-figures
```

Options:
- `--descriptive`: Run descriptive analysis
- `--synthetic`: Run synthetic prompt generation
- `--api-key`: Anthropic API key for synthetic prompts
- `--output-dir`: Directory to save analysis results
- `--sample-size`: Number of samples to use
- `--save-figures`: Save generated figures from descriptive analysis

### Using as a Python Package

```python
from courtpress.models import RuleBasedFilter, SemanticSimilarityAnalyzer
from courtpress.analysis import DescriptiveAnalyzer
from courtpress.data import CourtDataLoader

# Load data
loader = CourtDataLoader()
data = loader.load_data()

# Analyze data
analyzer = DescriptiveAnalyzer()
results = analyzer.run_analysis(df=data, save_figures=True)

# Apply filtering
rule_filter = RuleBasedFilter()
filtered_data = rule_filter.filter(data)
```

## Module Structure

```
src/
└── courtpress/
    ├── __init__.py           # Package initialization
    ├── data/                 # Data loading modules
    ├── preprocessing/        # Text preprocessing modules
    ├── models/               # Analysis models
    │   ├── rule_based.py            # Rule-based filtering
    │   ├── semantic_similarity.py   # Semantic similarity analysis
    │   ├── supervised_ml.py         # Supervised machine learning
    │   ├── clustering.py            # Unsupervised clustering
    │   ├── combined_methods.py      # Combined filtering methods
    │   └── synthetic_prompts.py     # Synthetic prompt generation
    ├── analysis/             # Analysis modules
    │   └── descriptive.py           # Descriptive analysis
    ├── utils/                # Utility functions
    └── scripts/              # Command-line scripts
        ├── run_cleaning.py           # Data cleaning script
        ├── run_analysis.py           # Descriptive analysis script
        ├── generate_prompts.py       # Synthetic prompt generation script
        └── run_combined_analysis.py  # Combined analysis script
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- transformers
- torch
- anthropic (optional, for synthetic prompt generation)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 