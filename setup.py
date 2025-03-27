from setuptools import setup, find_packages

setup(
    name="courtpress",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "spacy>=3.7.0",
        "transformers>=4.35.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "courtpress-analyze=courtpress.scripts.run_analysis:main",
            "courtpress-clean=courtpress.scripts.run_cleaning:main",
            "courtpress-prompts=courtpress.scripts.generate_prompts:main",
            "courtpress-combined=courtpress.scripts.run_combined_analysis:main",
        ],
    },
)
