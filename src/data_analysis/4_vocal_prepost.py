#!/usr/bin/env python
"""
Vocal feature analysis script for 5-MeO-DMT study.
Performs statistical analysis on vocal features pre/post intervention.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pathlib import Path

# Constants
CORE_DIR = Path('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals')
OUTPUT_DIR = CORE_DIR / 'outputs'
FIGURE_DIR = OUTPUT_DIR / 'figures'
TABLE_DIR = OUTPUT_DIR / 'tables'
PSYCHOMETRICS = ['survey_aPPS_total', 'survey_EBI', 'survey_ASC_OBN', 'survey_ASC_DED', 'survey_bSWEBWBS']


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


def create_all_features_plot(results_df, save_path):
    """Create a bar plot showing all vocal feature changes."""
    # Sort by effect size
    results_df = results_df.sort_values('Mean Difference')
    
    # Define a custom darker color palette for significant results
    sig_colors = [
        '#1f77b4',  # dark blue
        '#d62728',  # dark red
        '#2ca02c',  # dark green
        '#9467bd',  # dark purple
        '#8c564b',  # dark brown
        '#e377c2',  # dark pink
        '#7f7f7f',  # dark gray
    ]
    
    # Create plot with adjusted layout
    fig, ax = plt.subplots(figsize=(15, 2))
    fig.subplots_adjust(left=0.4, right=0.95, bottom=0.5, top=0.9)
    
    # Plot vertical bars
    x_pos = np.arange(len(results_df))
    bars = ax.bar(
        x_pos, 
        results_df['Mean Difference'],
        capsize=3,
        color=['lightgray' if float(p) >= 0.05 else sig_colors[i % len(sig_colors)] 
               for i, p in enumerate(results_df['p-adjusted'])],
        width=0.6,
        error_kw={'elinewidth': 1, 'capsize': 3, 'ecolor': 'dimgray'}
    )
    
    # Set x-axis labels with 45 degree rotation
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        results_df['Feature'], 
        rotation=45,
        ha='right',   
        fontsize=8
    )   
    
    # Color labels for significant features
    for i, (tick_label, p_adj) in enumerate(zip(ax.get_xticklabels(), results_df['p-adjusted'])):
        if float(p_adj) < 0.05:
            tick_label.set_color(sig_colors[i % len(sig_colors)])
        else:
            tick_label.set_color('gray')
    
    ax.tick_params(axis='x', which='major', pad=5)
    
    # Add significance stars
    for i, row in results_df.iterrows():
        p_adj = float(row['p-adjusted'])
        if p_adj < 0.05:  # only add stars for significant results
            if p_adj < 0.001:
                significance = '***'
            elif p_adj < 0.01:
                significance = '**'
            else:
                significance = '*'
            
            # Get the exact bar object and its properties
            bar = bars[i]
            x_pos = bar.get_x() + bar.get_width()/2  # Center of the bar
            bar_height = row['Mean Difference']
            
            # Position stars above or below bar depending on bar direction
            if bar_height >= 0:
                y_pos = bar_height + 0.05  # Place slightly above positive bars
            else:
                y_pos = bar_height - 0.05  # Place slightly below negative bars
            
            ax.text(x_pos, y_pos, significance,
                   ha='center',
                   va='bottom' if bar_height >= 0 else 'top',
                   fontsize=10,
                   color='black')
    
    # Final styling
    ax.set_ylim(-1, 1)
    ax.set_ylabel('Standardized Change in Vocal Features\n(Z-scores of Post minus Pre differences)', 
                 fontsize=10,
                 labelpad=10)
    
    # Save plot
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', pad_inches=0.5)
    plt.close()


def create_forest_plot(results_df, save_path):
    """Create a forest plot showing effect sizes and confidence intervals."""
    # Sort by effect size
    plot_df = results_df.sort_values('Mean Difference')
    
    fig, ax = plt.subplots(figsize=(10, len(plot_df)*0.3))
    
    # Plot points and CI lines
    y_pos = np.arange(len(plot_df))
    ax.scatter(plot_df['Mean Difference'], y_pos, c='black', zorder=3)
    
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 
                color='black', linewidth=1, zorder=2)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['Feature'])
    ax.set_xlabel('Standardized Mean Difference (95% CI)')
    
    # Add significance markers
    for i, p_adj in enumerate(plot_df['p-adjusted']):
        if float(p_adj) < 0.05:
            ax.text(1.1, i, '*'*sum([float(p_adj) < cutoff for cutoff in [0.05, 0.01, 0.001]]))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_combined_plot(df, results_df, save_path):
    """Create a combined plot with forest plot and box plots for significant features."""
    # Filter for significant results and ensure numeric types
    sig_df = results_df[results_df['p-adjusted'].astype(float) < 0.05].copy()
    sig_df['Mean Difference'] = sig_df['Mean Difference'].astype(float)
    sig_df['ci_lower'] = sig_df['ci_lower'].astype(float)
    sig_df['ci_upper'] = sig_df['ci_upper'].astype(float)
    sig_df = sig_df.sort_values('Mean Difference')
    
    # Create figure with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, len(sig_df)*0.4 + 1), 
                                  gridspec_kw={'width_ratios': [1.5, 1]})
    
    # Forest plot on the left (ax1)
    y_pos = np.arange(len(sig_df))
    
    for idx, row in sig_df.iterrows():
        color = '#2166AC' if row['Mean Difference'] < 0 else '#B2182B'
        
        # Plot CI lines and points
        ax1.plot([row['ci_lower'], row['ci_upper']], 
                [y_pos[sig_df.index.get_loc(idx)]] * 2,
                color=color, linewidth=2)
        ax1.scatter(row['Mean Difference'], 
                   y_pos[sig_df.index.get_loc(idx)],
                   color=color, s=100)
    
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sig_df['Feature'])
    ax1.set_xlabel('Standardized Mean Difference (95% CI)')
    
    # Box plots on the right (ax2)
    positions = np.arange(0, len(sig_df)*3, 3)
    
    for i, (_, row) in enumerate(sig_df.iterrows()):
        feature = row['feature']
        pre_data = df[f'{feature}_Pre']
        post_data = df[f'{feature}_Post']
        
        # Create box plots
        bp = ax2.boxplot([pre_data, post_data], 
                        positions=[positions[i], positions[i]+1],
                        widths=0.6,
                        showfliers=False)
        
        # Add individual participant lines
        for pre, post in zip(pre_data, post_data):
            ax2.plot([positions[i], positions[i]+1], [pre, post], 
                    color='gray', alpha=0.2, linewidth=0.5)
    
    ax2.set_xticks(positions + 0.5)
    ax2.set_xticklabels(['Pre/Post' for _ in range(len(sig_df))], rotation=45)
    ax2.set_ylabel('Raw Values')
    
    plt.suptitle('Significant Changes in Vocal Features (p < 0.05)', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_significant_changes_plot(results_df, save_path):
    """Create a plot focused on statistically significant changes."""
    # Filter for significant results
    sig_df = results_df[results_df['p-adjusted'].astype(float) < 0.05].copy()
    sig_df['Mean Difference'] = sig_df['Mean Difference'].astype(float)
    sig_df['ci_lower'] = sig_df['ci_lower'].astype(float)
    sig_df['ci_upper'] = sig_df['ci_upper'].astype(float)
    
    # Sort by absolute effect size
    sig_df = sig_df.sort_values('Mean Difference')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 0.8*len(sig_df)))
    
    # Create colors based on direction of effect
    colors = ['#2166AC' if val < 0 else '#B2182B' if val > 0 else 'gray' 
              for val in sig_df['Mean Difference']]
    
    # Plot horizontal bars
    y_pos = np.arange(len(sig_df))
    bars = ax.barh(y_pos, 
                  sig_df['Mean Difference'],
                  height=0.6,
                  color=colors)
    
    # Add error bars
    error_bars = ax.errorbar(sig_df['Mean Difference'],
                           y_pos,
                           xerr=[(sig_df['Mean Difference'] - sig_df['ci_lower']),
                                (sig_df['ci_upper'] - sig_df['Mean Difference'])],
                           fmt='none',
                           color='black',
                           capsize=3)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add significance markers
    for idx, p_val in enumerate(sig_df['p-adjusted'].astype(float)):
        stars = '*' * sum([p_val < cutoff for cutoff in [0.05, 0.01, 0.001]])
        x_pos = sig_df['Mean Difference'].iloc[idx]
        x_offset = 0.01 if x_pos >= 0 else -0.01
        ax.text(x_pos + x_offset, idx, 
                stars, 
                va='center',
                ha='left' if x_pos >= 0 else 'right')
    
    # Customize y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(['vocal_' + clean_feature_name(feat) for feat in sig_df['feature']])
    
    # Set reasonable x-axis limits
    max_abs = max(abs(sig_df['ci_lower'].min()), abs(sig_df['ci_upper'].max()))
    ax.set_xlim(-max_abs*1.2, max_abs*1.2)
    
    # Add labels and title
    ax.set_xlabel('Significant Changes in Vocal Features\n(Post-5-MeO-DMT minus Pre-5-MeO-DMT)')
    
    # Add grid
    ax.grid(True, axis='x', linestyle=':', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_significant_features(results_df):
    """Print significant features in an easy to copy-paste format."""
    # Convert to numeric p-adjusted
    results_df['p-adjusted'] = results_df['p-adjusted'].astype(float)
    significant_features = results_df[results_df['p-adjusted'] < 0.05]['feature'].tolist()
    
    # Print for manual labeling
    print("\nSignificant features for manual labeling:")
    print("\nsig_feature_labels = {")
    for feature in significant_features:
        print(f"    '{feature}': '',  # {results_df[results_df['feature'] == feature]['Mean Difference'].values[0]:.3f}")
    print("}")


def save_results_table(results_df):
    """Save the results table to CSV."""
    # Sort by adjusted p-value
    table_df = results_df.sort_values('raw_p')
    
    # Format p-values and mean difference for table
    for col in ['p-value', 'p-adjusted', 'Mean Difference']:
        table_df[col] = table_df[col].apply(lambda x: f"{float(x):.3f}")
    
    # Select and rename columns for final table
    table_df = table_df[[
        'Feature', 
        'Pre Mean (SD)', 
        'Post Mean (SD)', 
        'Mean Difference',
        '95% CI',
        't-statistic',
        'p-value',
        'p-adjusted',
        'Significance'
    ]].copy()
    
    # Create output directory if it doesn't exist
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    table_df.to_csv(TABLE_DIR / 'vocal_features_statistics.csv', index=False)
    
    # Display the table
    print("\nVocal Features Statistics Table:")
    print(table_df.to_string(index=False))


def main():
    """Main function to run the vocal feature analysis."""
    # Ensure output directories exist
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    df, vocal_features = load_and_preprocess_data()
    
    # Run statistical analysis
    results_df = run_statistical_analysis(df, vocal_features)
    
    # Create plots
    create_all_features_plot(results_df, FIGURE_DIR / 'vocal_changes_all.png')
    create_forest_plot(results_df, FIGURE_DIR / 'vocal_changes_forest.png')
    create_combined_plot(df, results_df, FIGURE_DIR / 'vocal_changes_combined.png')
    create_significant_changes_plot(results_df, FIGURE_DIR / 'vocal_changes_significant.png')
    
    # Print significant features
    print_significant_features(results_df)
    
    # Save results table
    save_results_table(results_df)


if __name__ == "__main__":
    main()
