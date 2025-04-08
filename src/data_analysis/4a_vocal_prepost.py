#!/usr/bin/env python
"""
Forest plot visualization for vocal feature changes in 5-MeO-DMT study.
This script creates a detailed forest plot showing standardized mean differences
in vocal features before and after intervention.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Constants
CORE_DIR = Path('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals')
OUTPUT_DIR = CORE_DIR / 'outputs'
FIGURE_DIR = OUTPUT_DIR / 'figures'


def load_and_preprocess_data():
    """Load and preprocess vocal feature data."""
    df = pd.read_csv(CORE_DIR / 'data/final/means_level+pca.csv')
    vocal_features = [col for col in df.columns if 'vocal_' in col]
    
    # Remove rows where any of the vocal features Pre is nan
    df = df[df[vocal_features].notna().all(axis=1)]
    
    print(f"\nTotal number of participants in analysis: {len(df)}")
    return df, vocal_features


def clean_feature_name(feature):
    """Clean feature names for better readability."""
    return feature.replace('vocal_', '')


def run_statistical_analysis(df, vocal_features):
    """Run t-tests on pre/post vocal features and apply FDR correction."""
    results = []
    
    for feature in vocal_features:
        feature_base = feature.replace('_Pre', '').replace('_Post', '')
        if '_Pre' in feature:  # only process each feature once
            # Get Pre and Post values
            pre_values = df[f'{feature_base}_Pre']
            post_values = df[f'{feature_base}_Post']
            
            # Calculate raw difference
            diff = post_values - pre_values
            
            # Calculate z-scores
            mean_diff = diff.mean()
            std_diff = diff.std()
            z_score = mean_diff / std_diff if std_diff != 0 else 0
            
            # Calculate standard error and CI for z-score
            std_err = 1 / np.sqrt(len(diff))  # SE of z-score
            ci_lower = z_score - 1.96 * std_err
            ci_upper = z_score + 1.96 * std_err
            
            # Calculate t-statistic and p-value
            t_stat, p_val = stats.ttest_rel(pre_values, post_values)
            
            results.append({
                'feature': feature_base,
                'Feature': clean_feature_name(feature_base),
                'Pre Mean (SD)': f"{pre_values.mean():.3f} ({pre_values.std():.3f})",
                'Post Mean (SD)': f"{post_values.mean():.3f} ({post_values.std():.3f})",
                'Mean Difference': z_score,
                'std_dev': 1.0,  # Since we're using z-scores, SD is 1
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                't-statistic': t_stat,
                'p-value': p_val,
                'raw_p': p_val
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    results_df['p-adjusted'] = multipletests(results_df['raw_p'], method='fdr_bh')[1]
    
    # Add significance markers
    results_df['Significance'] = ''
    results_df.loc[results_df['p-adjusted'] < 0.05, 'Significance'] = '*'
    results_df.loc[results_df['p-adjusted'] < 0.01, 'Significance'] = '**'
    results_df.loc[results_df['p-adjusted'] < 0.001, 'Significance'] = '***'
    
    return results_df


def create_comprehensive_forest_plot(results_df, save_path):
    """
    Create a detailed forest plot showing all vocal feature changes with
    standardized mean differences and confidence intervals.
    """
    # Sort by alphabetical order of feature names
    plot_df = results_df.sort_values('Mean Difference')
    
    # Calculate the figure height based on number of features - reduced scaling factor
    feature_count = len(plot_df)
    fig_height = max(6, feature_count * 0.15)  # Further reduced from 0.22 to 0.2
    # fig_height = 9
    
    # Create figure with appropriate dimensions
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Disable automatic layout adjustments
    fig.set_constrained_layout(False)
    fig.set_tight_layout(False)
    
    # More aggressive margin reduction, especially on top and bottom
    plt.subplots_adjust(left=0.4, right=0.9, top=0.99, bottom=0.03, hspace=0)
    
    # Plot points and CI lines
    y_pos = np.arange(len(plot_df))
    
    # Plot CI lines first (behind the points)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ci_lower = float(row['ci_lower']) if isinstance(row['ci_lower'], str) else row['ci_lower']
        ci_upper = float(row['ci_upper']) if isinstance(row['ci_upper'], str) else row['ci_upper']
        mean_diff = float(row['Mean Difference']) if isinstance(row['Mean Difference'], str) else row['Mean Difference']
        
        ax.plot([ci_lower, ci_upper], [i, i], 
                color='black', linewidth=1, zorder=1)
    
    # Plot the points on top of the lines
    scatter = ax.scatter(
        plot_df['Mean Difference'].astype(float), 
        y_pos,
        c='black', 
        s=20, 
        zorder=2
    )
    
    # Add vertical line at zero
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, zorder=0)
    
    # Add vertical grid lines
    ax.grid(axis='x', linestyle=':', alpha=0.3, zorder=0)
    
    # Customize axes - reduce font size to allow more compact presentation
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['Feature'], fontsize=8)

    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    
    # Add significance markers at the right edge
    for i, (_, row) in enumerate(plot_df.iterrows()):
        p_adj = float(row['p-adjusted']) if isinstance(row['p-adjusted'], str) else row['p-adjusted']
        if p_adj < 0.05:
            ax.annotate('*', xy=(1.02, i), xycoords=('axes fraction', 'data'), 
                        fontsize=10, ha='center', va='center')
    
    # Set x-axis limits to ensure symmetry around zero and space for significance markers
    max_abs_ci = max(
        abs(plot_df['ci_lower'].astype(float).min()),
        abs(plot_df['ci_upper'].astype(float).max())
    )
    ax.set_xlim(-max_abs_ci * 1.1, max_abs_ci * 1.1)
    
    # Add axis labels with minimal padding
    ax.set_xlabel('Pre to Post 5-MeO-DMT \nStandardized Mean Difference (95% CI)', fontsize=10, labelpad=2)
    
    # Remove unnecessary padding from y-axis
    ax.tick_params(axis='y', pad=1)
    ax.tick_params(axis='x', pad=2)
    
    # Create output directory if it doesn't exist
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save figure with high resolution - avoid using tight_layout which adds padding
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


def print_significant_features(results_df):
    """Print a list of all significant features after FDR correction."""
    # Get only significant features (p-adjusted < 0.05)
    sig_features = results_df[results_df['p-adjusted'] < 0.05]
    
    # Sort by significance level and then by effect size magnitude
    sig_features = sig_features.sort_values(
        by=['p-adjusted', 'Mean Difference'], 
        ascending=[True, False]
    )
    
    # Print the list of significant features
    print("\nSignificant vocal features after FDR correction:")
    print("="*60)
    print(f"{'Feature':<25} {'Mean Diff':<10} {'p-adj':<10} {'Significance':<10}")
    print("-"*60)
    
    for _, row in sig_features.iterrows():
        print(f"{row['Feature']:<25} {row['Mean Difference']:.3f}      {row['p-adjusted']:.4f}    {row['Significance']}")
    
    print("="*60)
    print(f"Total significant features: {len(sig_features)}/{len(results_df)}")
    
    return sig_features


def create_significant_features_forest_plot(results_df, save_path):
    """
    Create an extremely compact forest plot showing only significant vocal features
    with minimal whitespace, but adequate padding for the x-axis label.
    """
    # Get only significant features
    sig_df = results_df[results_df['p-adjusted'] < 0.05].copy()
    
    if len(sig_df) == 0:
        print("No significant features found. Cannot create significant features plot.")
        return
    
    # Sort by mean difference
    sig_df = sig_df.sort_values('Mean Difference')
    
    # Create a dictionary mapping original feature names to human-readable labels
    human_readable_labels = {
        'shimmerLocaldB_sma3nz_amean': 'Avg Local Shimmer (dB, smoothed)',
        'jitterLocal_sma3nz_amean': 'Avg Local Jitter (smoothed)',
        'jitterLocal_sma3nz_stddevNorm': 'Normalized SD of Local Jitter (smoothed)'
    }
    
    # Replace feature names with human-readable labels
    sig_df_display = sig_df.copy()
    sig_df_display['Feature'] = sig_df_display['Feature'].map(
        lambda x: human_readable_labels.get(x, x)
    )
    
    # Create extremely compact figure with slightly more height for bottom spacing
    fig, ax = plt.subplots(figsize=(7, 2.0))
    
    # Disable automatic layout adjustments
    fig.set_constrained_layout(False)
    fig.set_tight_layout(False)
    
    # Ultra tight margins but with more bottom padding
    plt.subplots_adjust(left=0.45, right=0.95, top=0.8, bottom=0.4)  # Increased left from 0.3 to 0.45 for longer labels
    
    # Plot positions
    y_positions = range(len(sig_df_display))
    
    # Plot CI lines
    for i, (_, row) in enumerate(sig_df.iterrows()):
        ci_lower = float(row['ci_lower']) if isinstance(row['ci_lower'], str) else row['ci_lower']
        ci_upper = float(row['ci_upper']) if isinstance(row['ci_upper'], str) else row['ci_upper']
        
        ax.plot([ci_lower, ci_upper], [i, i], 
                color='black', linewidth=1, zorder=1)
    
    # Plot points
    scatter = ax.scatter(
        sig_df['Mean Difference'].astype(float), 
        y_positions,
        c='black', 
        s=15,
        zorder=2
    )
    
    # Add vertical line at zero
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, zorder=0)
    
    # Sparser grid lines
    ax.grid(axis='x', linestyle=':', alpha=0.3, zorder=0)
    
    # Customize axes with smaller font
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sig_df_display['Feature'], fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    
    # Slightly looser y-axis limits - add more padding at bottom
    ax.set_ylim(-0.3, len(sig_df) - 0.1)
    
    # Add significance markers
    for i, (_, row) in enumerate(sig_df.iterrows()):
        ax.annotate(row['Significance'], xy=(1.01, i), xycoords=('axes fraction', 'data'), 
                    fontsize=8, ha='left', va='center')
    
    # Set x-axis limits
    max_abs_ci = max(
        abs(sig_df['ci_lower'].astype(float).min()),
        abs(sig_df['ci_upper'].astype(float).max())
    )
    ax.set_xlim(-max_abs_ci * 1.05, max_abs_ci * 1.05)
    
    # Smaller x-axis label with slightly more padding
    ax.set_xlabel('Pre to Post 5-MeO-DMT\nStandardized Mean Difference (95% CI)', 
                 fontsize=8, labelpad=4)
    
    # Minimal tick padding
    ax.tick_params(axis='y', pad=0.5)
    ax.tick_params(axis='x', pad=1.5)
    
    # Save with more padding at the bottom
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()


def save_results_table(results_df, n_observations):
    """Save the results table to CSV with specified column format."""
    # Sort by adjusted p-value
    table_df = results_df.sort_values('raw_p')
    
    # Calculate standard error (SE = SD/sqrt(n))
    table_df['Standard Error'] = 1.0 / np.sqrt(n_observations)  # Since we're using z-scores, SD is 1
    
    # Format values for table
    table_df['Pre Mean'] = table_df['Pre Mean (SD)'].apply(lambda x: float(x.split(' ')[0]))
    table_df['Post Mean'] = table_df['Post Mean (SD)'].apply(lambda x: float(x.split(' ')[0]))
    
    # Select and rename columns for final table
    table_df = table_df[[
        'Feature', 
        'Pre Mean',
        'Post Mean',
        'Mean Difference',
        'Standard Error',
        'ci_lower',
        'ci_upper',
        'raw_p',
        'p-adjusted',
        'Significance'
    ]].copy()
    
    # Rename columns to match requested format
    table_df = table_df.rename(columns={
        'ci_lower': 'CI Lower',
        'ci_upper': 'CI Upper',
        'raw_p': 'P-Value',
        'p-adjusted': 'Corrected P-Value'
    })
    
    # Add N Observations column
    table_df['N Observations'] = n_observations
    
    # Format numeric columns
    for col in ['Pre Mean', 'Post Mean', 'Mean Difference', 'Standard Error', 
                'CI Lower', 'CI Upper', 'P-Value', 'Corrected P-Value']:
        table_df[col] = table_df[col].apply(lambda x: f"{float(x):.3f}")
    
    # Create output directory if it doesn't exist
    TABLE_DIR = OUTPUT_DIR / 'tables'
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    table_df.to_csv(TABLE_DIR / 'vocal_features_statistics.csv', index=False)
    
    # Display the table
    print("\nVocal Features Statistics Table:")
    print(table_df.to_string(index=False))


def main():
    """Main function to run the forest plot visualization."""
    # Ensure output directory exists
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    df, vocal_features = load_and_preprocess_data()
    
    print(vocal_features)
    # Run statistical analysis
    results_df = run_statistical_analysis(df, vocal_features)
    
    # Print significant features as a list
    sig_features_df = print_significant_features(results_df)
    
    # Create comprehensive forest plot
    create_comprehensive_forest_plot(
        results_df, 
        FIGURE_DIR / 'vocal_features_forest_plot.png'
    )
    
    # Create forest plot with only significant features
    create_significant_features_forest_plot(
        results_df,
        FIGURE_DIR / 'vocal_features_significant_forest_plot.png'
    )
    
    # Save results table
    save_results_table(results_df, len(df))
    
    print("Forest plots and statistics table have been created and saved successfully.")


if __name__ == "__main__":
    main()
