#!/usr/bin/env python
"""
Analysis script for correlations between PCA components and LIWC features
in 5-MeO-DMT study data.
"""

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import numpy as np
import json
from pathlib import Path


def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    # Define constants
    core_dir = Path('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals')
    
    # Load data
    df = pd.read_csv(core_dir / 'data/final/means_level+pca.csv')
    print(f"Initial data shape: {df.shape}")
    
    # Define feature categories
    vocal_features = [col for col in df.columns if 'vocal_' in col and 'Pre' in col]
    
    # Remove rows where any of the vocal features Pre is nan
    df = df[df[vocal_features].notna().all(axis=1)]
    print(f"Data shape after filtering NaN values: {df.shape}")
    
    return df, core_dir


def load_liwc_labels(core_dir):
    """Load LIWC labels from config file."""
    with open(core_dir / 'config/liwc_labels.json', 'r') as f:
        liwc_labels = json.load(f)
    
    # Create flat dictionary of all LIWC labels
    flat_labels = {}
    for category in liwc_labels.values():
        flat_labels.update(category)
    
    return flat_labels


def filter_liwc_features(df, flat_labels):
    """Filter LIWC features that have matching labels."""
    # Get all LIWC features from the dataset
    liwc_features = [col for col in df.columns if 'liwc_' in col and 'Pre' in col]
    
    # Filter for only Pre-session LIWC features that have matching labels
    valid_liwc_features = []
    for col in liwc_features:
        base_feature = col.replace('liwc_', '').replace('_Pre', '')
        if base_feature in flat_labels:
            valid_liwc_features.append(col)
        else:
            print(f"Excluding {col} - no matching label in config")
    
    return liwc_features, valid_liwc_features


def analyze_correlations(df, target_variables, feature_variables, flat_labels):
    """Analyze correlations between target and feature variables."""
    for target in target_variables:
        # Initialize storage for results
        correlations = {}
        p_values = {}
        valid_features = []
        
        for feature in feature_variables:
            # Get label for the feature
            if 'liwc_' in feature:
                base_feature = feature.replace('liwc_', '').replace('_Pre', '')
                if base_feature not in flat_labels:
                    continue
                label = flat_labels[base_feature]
            elif 'vocal_' in feature:
                label = feature
            
            # Check for constant values
            if df[feature].nunique() <= 1:
                print(f"Warning: {feature} ({label}) has constant values")
                continue
            
            # Check for missing values
            if df[feature].isna().any() or df[target].isna().any():
                print(f"Warning: {feature} ({label}) contains missing values")
                continue
            
            # Calculate correlation
            corr, p_val = stats.pearsonr(df[target], df[feature])
            correlations[feature] = corr
            p_values[feature] = p_val
            valid_features.append(feature)
        
        # Apply multiple comparisons correction and print results
        print_correlation_results(target, valid_features, correlations, p_values, flat_labels)


def print_correlation_results(target, valid_features, correlations, p_values, flat_labels):
    """Print the correlation results with FDR correction."""
    if not valid_features:
        print("\nNo valid features found for correlation analysis")
        return
    
    # Apply multiple comparisons correction
    rejected, p_adjusted = multipletests(
        list(p_values.values()), 
        method='fdr_bh'
    )[0:2]
    
    print(f"\nSignificant correlations with {target} (FDR-corrected p < 0.05):")
    
    # Print positive correlations
    print("\nPositive correlations:")
    for feature, corr, p_val, p_adj, is_rejected in zip(
            valid_features, 
            [correlations[f] for f in valid_features],
            [p_values[f] for f in valid_features],
            p_adjusted, 
            rejected):
        
        if is_rejected and corr > 0:
            base_feature = feature.replace('liwc_', '').replace('_Pre', '')
            if 'liwc_' in feature:
                label = flat_labels.get(base_feature, feature)
            else:
                label = feature
            print(f"{label}: r = {corr:.3f}, p = {p_val:.3f}, p_adj = {p_adj:.3f}")
    
    # Print negative correlations
    print("\nNegative correlations:")
    for feature, corr, p_val, p_adj, is_rejected in zip(
            valid_features, 
            [correlations[f] for f in valid_features],
            [p_values[f] for f in valid_features],
            p_adjusted, 
            rejected):
        
        if is_rejected and corr < 0:
            base_feature = feature.replace('liwc_', '').replace('_Pre', '')
            if 'liwc_' in feature:
                label = flat_labels.get(base_feature, feature)
            else:
                label = feature
            print(f"{label}: r = {corr:.3f}, p = {p_val:.3f}, p_adj = {p_adj:.3f}")


def main():
    """Main function to run the correlation analysis."""
    # Load and preprocess data
    df, core_dir = load_and_preprocess_data()
    
    # Load LIWC labels
    flat_labels = load_liwc_labels(core_dir)
    
    # Filter LIWC features
    liwc_features, valid_liwc_features = filter_liwc_features(df, flat_labels)
    
    # Define target and feature variables
    target_variables = ['PC1', 'PC2', 'PC3']
    feature_variables = liwc_features  # Using all LIWC features
    
    # Analyze correlations
    analyze_correlations(df, target_variables, feature_variables, flat_labels)


if __name__ == "__main__":
    main()
