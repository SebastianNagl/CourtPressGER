import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from matplotlib_venn import venn3
from collections import Counter


def plot_rule_based_criteria(df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot analysis of rule-based filtering criteria.

    Args:
        df: DataFrame with rule-based filtering results
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure with visualizations
    """
    # Create figure
    fig = plt.figure(figsize=figsize)

    # Required columns for this visualization
    required_cols = [
        'filter_future_indicators',
        'filter_date_patterns',
        'filter_headline_indicators',
        'filter_headline_date'
    ]

    # Verify that required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"DataFrame is missing required columns: {missing_cols}")

    # 1. Barplot for individual criteria
    criterion_counts = pd.DataFrame({
        'Kriterium': [
            'Zukunftsindikatoren',
            'Datumsmuster',
            'Überschriften-Indikatoren',
            'Überschrift + Datum'
        ],
        'Anzahl': [
            df['filter_future_indicators'].sum(),
            df['filter_date_patterns'].sum(),
            df['filter_headline_indicators'].sum(),
            df['filter_headline_date'].sum()
        ]
    })

    criterion_counts['Prozent'] = criterion_counts['Anzahl'] / len(df) * 100

    plt.subplot(2, 2, 1)
    sns.barplot(data=criterion_counts, x='Kriterium', y='Prozent')
    plt.xticks(rotation=45, ha='right')
    plt.title('Anteil der gefilterten Einträge pro Kriterium')
    plt.ylabel('Prozent der Einträge')

    # 2. Venn diagram for overlaps
    plt.subplot(2, 2, 2)

    # Create sets for the three main criteria
    future_set = set(df[df['filter_future_indicators']].index)
    date_set = set(df[df['filter_date_patterns']].index)
    headline_set = set(df[df['filter_headline_indicators']].index)

    venn3([future_set, date_set, headline_set],
          ('Zukunftsindikatoren', 'Datumsmuster', 'Überschriften'))
    plt.title('Überlappung der Filterkriterien')

    # 3. Stacked bar chart for combined effects
    plt.subplot(2, 2, 3)
    filter_combinations = df[['filter_future_indicators',
                              'filter_date_patterns', 'filter_headline_indicators']].sum(axis=1)
    combination_counts = filter_combinations.value_counts().sort_index()
    plt.bar(range(len(combination_counts)), combination_counts)
    plt.xticks(range(len(combination_counts)), [
               f"{i} Kriterien" for i in combination_counts.index])
    plt.title('Anzahl erfüllter Kriterien pro Eintrag')
    plt.ylabel('Anzahl Einträge')

    # 4. Summary statistics
    plt.subplot(2, 2, 4)
    summary_stats = pd.DataFrame({
        'Metrik': [
            'Gesamtanzahl Einträge',
            'Mind. 1 Kriterium erfüllt',
            'Mind. 2 Kriterien erfüllt',
            'Alle Kriterien erfüllt'
        ],
        'Anzahl': [
            len(df),
            (filter_combinations >= 1).sum(),
            (filter_combinations >= 2).sum(),
            (filter_combinations == 3).sum()
        ]
    })
    summary_stats['Prozent'] = summary_stats['Anzahl'] / len(df) * 100

    plt.axis('off')
    plt.table(cellText=summary_stats.values,
              colLabels=summary_stats.columns,
              cellLoc='center',
              loc='center',
              bbox=[0, 0, 1, 1])
    plt.title('Zusammenfassende Statistik')

    plt.tight_layout()

    return fig


def plot_similarity_distribution(df: pd.DataFrame,
                                 similarity_col: str = 'tfidf_similarity',
                                 threshold: Optional[float] = None,
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the distribution of similarity scores with threshold line.

    Args:
        df: DataFrame with similarity scores
        similarity_col: Column name with similarity scores
        threshold: Optional threshold value to mark on the plot
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure with histogram
    """
    # Check if column exists
    if similarity_col not in df.columns:
        raise ValueError(f"Column '{similarity_col}' not found in DataFrame")

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Plot histogram with KDE
    sns.histplot(df[similarity_col], bins=50, kde=True)
    plt.title('Verteilung der Ähnlichkeitswerte')
    plt.xlabel('Ähnlichkeit')
    plt.ylabel('Anzahl der Einträge')

    # Add threshold line if provided
    if threshold is None and len(df) > 0:
        # Default to 10th percentile if not provided
        threshold = df[similarity_col].quantile(0.1)

    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--',
                    label=f'Schwellenwert: {threshold:.3f}')
        plt.legend()

    plt.grid(True, alpha=0.3)

    return fig


def plot_classifier_results(training_stats: Dict[str, Any], figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Visualize the results of the supervised classifier.

    Args:
        training_stats: Dictionary with training statistics
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure with visualizations
    """
    # Create figure
    fig = plt.figure(figsize=figsize)

    # 1. Confusion matrix
    if 'confusion_matrix' in training_stats:
        plt.subplot(2, 2, 1)
        cm = training_stats['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0.5, 1.5], ['Relevant', 'Irrelevant'])
        plt.yticks([0.5, 1.5], ['Relevant', 'Irrelevant'])

    # 2. Classification report metrics
    if 'classification_report' in training_stats:
        plt.subplot(2, 2, 2)
        cr = training_stats['classification_report']

        metrics = ['precision', 'recall', 'f1-score']
        classes = ['0', '1']  # Relevant (0) and Irrelevant (1)

        # Extract metrics for each class
        metric_data = {
            'Class': ['Relevant', 'Irrelevant', 'Weighted Avg'],
            'Precision': [cr['0']['precision'], cr['1']['precision'], cr['weighted avg']['precision']],
            'Recall': [cr['0']['recall'], cr['1']['recall'], cr['weighted avg']['recall']],
            'F1-Score': [cr['0']['f1-score'], cr['1']['f1-score'], cr['weighted avg']['f1-score']]
        }

        # Create a DataFrame
        metrics_df = pd.DataFrame(metric_data)

        # Plot metrics
        metrics_df_melted = pd.melt(metrics_df, id_vars=['Class'], value_vars=[
                                    'Precision', 'Recall', 'F1-Score'])
        sns.barplot(data=metrics_df_melted, x='Class',
                    y='value', hue='variable')
        plt.title('Classification Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend(title='Metric')

    # 3. Class distribution
    if 'train_class_dist' in training_stats:
        plt.subplot(2, 2, 3)
        class_dist = training_stats['train_class_dist']
        plt.pie([class_dist[0], class_dist[1]],
                labels=['Relevant', 'Irrelevant'],
                autopct='%1.1f%%',
                colors=['lightblue', 'salmon'])
        plt.title('Class Distribution in Training Data')

    # 4. Summary statistics
    plt.subplot(2, 2, 4)
    summary_data = []

    if 'train_size' in training_stats:
        summary_data.append(['Training samples', training_stats['train_size']])

    if 'test_size' in training_stats:
        summary_data.append(['Test samples', training_stats['test_size']])

    if 'pos_ratio' in training_stats:
        summary_data.append(
            ['Irrelevant ratio', f"{training_stats['pos_ratio']:.2%}"])

    if 'classification_report' in training_stats:
        summary_data.append(
            ['Accuracy', f"{training_stats['classification_report']['accuracy']:.2%}"])

    plt.axis('off')

    if summary_data:
        plt.table(cellText=summary_data,
                  colLabels=['Metric', 'Value'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])

    plt.title('Model Summary')

    plt.tight_layout()

    return fig


def plot_method_comparison(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create visualizations comparing different filtering methods.

    Args:
        df: DataFrame with filtering results
        figsize: Figure size (width, height)

    Returns:
        Tuple of Matplotlib figures (bar chart, Venn diagram)
    """
    # Required columns for comparison
    method_columns = {
        'Rule-based': 'is_announcement_rule',
        'Similarity': 'is_dissimilar_tfidf',
        'ML': 'is_irrelevant_ml',
        'Clustering': 'is_irrelevant_cluster',
        'Combined': 'is_irrelevant_combined'
    }

    # Check which methods are available
    available_methods = {name: col for name,
                         col in method_columns.items() if col in df.columns}

    if not available_methods:
        raise ValueError("No filtering method columns found in DataFrame")

    # 1. Create bar chart comparison
    bar_fig = plt.figure(figsize=figsize)

    # Calculate counts and percentages
    method_stats = []
    for name, col in available_methods.items():
        count = df[col].sum()
        percent = count / len(df) * 100
        method_stats.append({
            'Method': name,
            'Count': count,
            'Percent': percent
        })

    # Create DataFrame for plotting
    stats_df = pd.DataFrame(method_stats)

    # Plot bar chart
    plt.subplot(1, 2, 1)
    sns.barplot(data=stats_df, x='Method', y='Count')
    plt.title('Irrelevant Entries by Method (Count)')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    sns.barplot(data=stats_df, x='Method', y='Percent')
    plt.title('Irrelevant Entries by Method (Percent)')
    plt.xticks(rotation=45)
    plt.ylabel('Percent of Dataset')

    plt.tight_layout()

    # 2. Create Venn diagram for the first three methods
    venn_fig = plt.figure(figsize=figsize)

    # Select up to three methods for Venn diagram
    venn_methods = list(available_methods.items())[:3]

    if len(venn_methods) > 0:
        sets = []
        labels = []

        for name, col in venn_methods:
            # Convert boolean column to set of indices
            if col == 'is_irrelevant_ml':
                # Special handling for ML column which might be integer
                method_set = set(df[df[col] == 1].index)
            else:
                method_set = set(df[df[col]].index)

            sets.append(method_set)
            labels.append(name)

        venn3(sets, labels)
        plt.title('Overlap of Irrelevant Entries Across Methods')

    return bar_fig, venn_fig
