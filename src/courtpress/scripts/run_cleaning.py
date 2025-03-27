#!/usr/bin/env python3
"""
Court Press Relevance Filtering

This script performs multi-method filtering of court press releases to identify
and remove press releases that don't directly relate to court decisions.

Usage:
    python run_cleaning.py [--gpu] [--save-figures] [--sample-size SAMPLE_SIZE]

Options:
    --gpu               Use GPU acceleration if available
    --save-figures      Save visualization figures
    --sample-size       Number of samples to use (default: use all)
"""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Import CourtPress modules
from courtpress.data.loader import CourtDataLoader
from courtpress.preprocessing.text_processor import TextProcessor
from courtpress.models.rule_based import RuleBasedFilter
from courtpress.models.semantic_similarity import SemanticSimilarityAnalyzer
from courtpress.models.supervised_ml import SupervisedClassifier
from courtpress.models.clustering import UnsupervisedClustering
from courtpress.models.combined_methods import CombinedFilter
from courtpress.utils.gpu_utils import check_gpu_availability, setup_gpu_environment, cleanup_gpu_memory
from courtpress.utils.visualization import (
    plot_rule_based_criteria,
    plot_similarity_distribution,
    plot_classifier_results,
    plot_method_comparison
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Court Press Relevance Filtering')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--save-figures', action='store_true',
                        help='Save visualization figures')
    parser.add_argument('--sample-size', type=int, default=0,
                        help='Number of samples to use (default: use all)')
    return parser.parse_args()


def save_figure(fig, filename, output_dir='figures'):
    """Save a figure to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")


def main():
    """Run the complete cleaning pipeline."""
    # Parse command line arguments
    args = parse_arguments()

    # Check GPU availability if requested
    use_gpu = False
    if args.gpu:
        gpu_available, gpu_info = check_gpu_availability()
        if gpu_available:
            use_gpu = True
            print("GPU acceleration enabled")
            setup_gpu_environment()
        else:
            print("GPU acceleration requested but not available, using CPU mode")
            print(f"GPU error: {gpu_info.get('error', 'Unknown error')}")

    # Load data
    print("\n--- Loading Data ---")
    loader = CourtDataLoader()
    df = loader.load_data()

    # Use sample if requested
    if args.sample_size > 0:
        df = loader.get_sample(n=args.sample_size)
        print(f"Using sample of {len(df)} entries")

    # Preprocess text
    print("\n--- Preprocessing Text ---")
    processor = TextProcessor(use_gpu=use_gpu)
    df = processor.prepare_dataset(df)

    # 1. Rule-based filtering
    print("\n--- Rule-based Filtering ---")
    rule_filter = RuleBasedFilter()
    df = rule_filter.apply_to_dataframe(df)

    if args.save_figures:
        fig = plot_rule_based_criteria(df)
        save_figure(fig, 'rule_based_criteria.png')

    # 2. Semantic similarity analysis
    print("\n--- Semantic Similarity Analysis ---")
    similarity_analyzer = SemanticSimilarityAnalyzer(use_gpu=use_gpu)
    df = similarity_analyzer.analyze_dataframe(df)

    if args.save_figures:
        fig = plot_similarity_distribution(df)
        save_figure(fig, 'similarity_distribution.png')

    # Clean up GPU memory if using GPU
    if use_gpu:
        cleanup_gpu_memory()

    # 3. Supervised ML classification
    print("\n--- Supervised ML Classification ---")
    classifier = SupervisedClassifier(use_gpu=use_gpu)
    df = classifier.analyze_dataframe(df)

    if args.save_figures and classifier.training_stats:
        fig = plot_classifier_results(classifier.training_stats)
        save_figure(fig, 'classifier_results.png')

    # Clean up GPU memory if using GPU
    if use_gpu:
        cleanup_gpu_memory()

    # 4. Unsupervised clustering
    print("\n--- Unsupervised Clustering ---")
    clustering = UnsupervisedClustering(use_gpu=use_gpu)

    # Get the TF-IDF vectors from the classifier
    if hasattr(classifier, 'vectorizer') and classifier.vectorizer is not None:
        vectors = classifier.vectorizer.transform(df['summary'].fillna(''))
        feature_names = classifier.vectorizer.get_feature_names_out()

        df = clustering.analyze_dataframe(df, vectors, feature_names)

        if args.save_figures:
            # Get reduced vectors for visualization
            vectors_reduced = clustering.fit_transform(vectors)
            cluster_labels = clustering.cluster(vectors_reduced)

            fig = clustering.visualize_clusters(
                vectors_reduced, cluster_labels)
            save_figure(fig, 'clustering_results.png')
    else:
        print("Skipping clustering as vectorizer is not available")

    # Clean up GPU memory if using GPU
    if use_gpu:
        cleanup_gpu_memory()

    # 5. Combined approach
    print("\n--- Combined Filtering Approach ---")
    combined_filter = CombinedFilter(vote_threshold=2)
    df = combined_filter.combine_methods(df)

    # Generate comparison table
    comparison_table = combined_filter.get_comparison_table(df)
    print("\nMethod Comparison:")
    print(comparison_table.to_string(index=False))

    if args.save_figures:
        bar_fig, venn_fig = plot_method_comparison(df)
        save_figure(bar_fig, 'method_comparison_bar.png')
        save_figure(venn_fig, 'method_comparison_venn.png')

    # Save results
    print("\n--- Saving Results ---")
    output_files = combined_filter.save_results(df)

    # Print results
    print("\n--- Results Summary ---")
    irrelevant_count = df['is_irrelevant_combined'].sum()
    print(f"Original dataset: {len(df)} entries")
    print(
        f"Identified as irrelevant: {irrelevant_count} entries ({irrelevant_count/len(df)*100:.2f}%)")
    print(
        f"Cleaned dataset: {len(df) - irrelevant_count} entries ({(len(df) - irrelevant_count)/len(df)*100:.2f}%)")

    print("\nOutput files:")
    for name, path in output_files.items():
        print(f"- {name}: {path}")

    print("\nCleaning process completed successfully!")


if __name__ == "__main__":
    main()
