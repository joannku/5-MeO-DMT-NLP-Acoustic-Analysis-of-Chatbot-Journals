import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
import json
import warnings

def filter_df(df):
    # Get means of numeric columns only
    means = df.select_dtypes(include=['float64', 'int64']).mean()
    
    # Print all column means for debugging
    print("\nColumn means:")
    for col in means.index:
        if col.startswith('goemo_'):
            print(f"{col}: {means[col]:.4f}")
    
    # Get all GoEmotion columns
    goemo_cols = [col for col in df.columns if col.startswith('goemo_') and col in means.index]
    
    print(f"\nFound {len(goemo_cols)} GoEmotion columns:")
    for col in goemo_cols:
        print(f"- {col}: {means[col]:.4f}")
    
    # Keep non-GoEmotion columns and GoEmotion columns
    non_goemo_cols = [col for col in df.columns if not col.startswith('goemo_')]
    df = df[non_goemo_cols + goemo_cols]

    # Remove participant with Participant ID = PRO71
    df = df[df['UserID'] != 'PRO71']

    # only keep columns that are in -14, 15 range for RelativeDate
    df = df[(df['RelativeDate'] >= -14) & (df['RelativeDate'] <= 14)]
    
    return df, goemo_cols

def run_mixed_effect_models(df):
    results_dict = {}
    p_values = []
    convergence_warnings = 0

    # Get all GoEmotion columns from the filtered dataframe
    goemo_categories = [col.replace('goemo_', '') for col in df.columns if col.startswith('goemo_')]
    
    if not goemo_categories:
        print("Warning: No GoEmotion categories found in the dataframe")
        return results_dict, p_values
    
    # Reset index to ensure proper alignment
    df = df.reset_index(drop=True)
    
    print(f"Processing {len(goemo_categories)} GoEmotion categories...")
    
    for category in goemo_categories:
        # Get the full column name
        goemo_col = f"goemo_{category}"
            
        try:
            # Create a clean subset of data for this category
            subset = df[[goemo_col, 'PrePost', 'UserID']].copy()
            subset = subset.dropna()
            subset = subset.reset_index(drop=True)
            
            if len(subset) == 0:
                print(f"Warning: No valid data for {category}, skipping...")
                continue
            
            # Try different optimization methods if default fails
            methods = ['nm', 'bfgs', 'lbfgs']
            result = None
            warning_caught = False
            
            for method in methods:
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        
                        model = smf.mixedlm(f"{goemo_col} ~ PrePost", subset, groups=subset['UserID'])
                        result = model.fit(method=method, maxiter=2000)
                        print(result.summary())
                        if any("ConvergenceWarning" in str(warn.message) for warn in w):
                            warning_caught = True
                            if method == methods[-1]:  # If this was the last method
                                convergence_warnings += 1
                                print(f"Warning: Convergence issues for {category} with all methods")
                            continue
                        break  # If no warning, keep this result
                        
                except Exception as e:
                    if method == methods[-1]:  # If this was the last method
                        raise e
                    continue
            
            coef = result.params["PrePost"]
            p_value = result.pvalues["PrePost"]
            ci_lower, ci_upper = result.conf_int().loc["PrePost"]
            
            p_values.append(p_value)
            
            # Store results using the original category name (without goemo_ prefix)
            results_dict[category] = {
                "Coefficient": coef,
                "P-Value": p_value,
                "Confidence Interval": (ci_lower, ci_upper),
                "Std. Error": result.bse["PrePost"],
                "Num. Observations": result.nobs,
                "Convergence Warning": warning_caught
            }
            
            print(f"Successfully processed {category} with {result.nobs} observations")
            
        except Exception as e:
            print(f"Error processing {category}: {str(e)}")
            continue

    print(f"\nConvergence warnings in {convergence_warnings} out of {len(goemo_categories)} models")
    return results_dict, p_values

def apply_p_value_correction(results_dict, p_values, alpha=0.05):
    if not p_values:
        print("Warning: No p-values to correct. Returning original results.")
        return results_dict
        
    corrected_p_values = smm.multipletests(p_values, alpha=alpha, method='fdr_bh')[1]

    # Add corrected p-values to the results dictionary
    for i, category in enumerate(results_dict.keys()):
        results_dict[category]["Corrected P-Value"] = corrected_p_values[i]

    return results_dict

def plot_mixed_effects_results(results_dict, filename='mixed_effects_results.png'):
    # Define a custom darker color palette
    custom_colors = [
        '#1f77b4',  # dark blue
        '#d62728',  # dark red
        '#2ca02c',  # dark green
        '#9467bd',  # dark purple
        '#8c564b',  # dark brown
        '#e377c2',  # dark pink
        '#7f7f7f',  # dark gray
        '#bcbd22',  # dark yellow
        '#17becf',  # dark cyan
        '#ff7f0e',  # dark orange
        '#1a5a89',  # another dark blue
        '#a52a2a',  # another dark red
        '#006400',  # another dark green
        '#4b0082',  # another dark purple
        '#8b4513',  # another dark brown
        '#c71585',  # another dark pink
        '#696969',  # another dark gray
        '#b8860b',  # another dark yellow
        '#008b8b',  # another dark cyan
        '#d2691e',  # another dark orange
        '#000080',  # navy
        '#800000',  # maroon
        '#006400',  # dark green
        '#4b0082',  # indigo
        '#8b4513',  # saddle brown
        '#800080',  # purple
        '#2f4f4f',  # dark slate gray
        '#8b008b',  # dark magenta
    ]
    
    # Ensure we have enough colors
    while len(custom_colors) < len(results_dict):
        custom_colors.extend(custom_colors)
    custom_colors = custom_colors[:len(results_dict)]
    
    color_dict = dict(zip(results_dict.keys(), custom_colors))

    # Sort results by coefficient value
    sorted_results = sorted(
        results_dict.items(),
        key=lambda x: x[1]['Coefficient'],
        reverse=True
    )

    # Prepare plotting data
    labels = [item[0] for item in sorted_results]
    coefficients = [item[1]['Coefficient'] for item in sorted_results]
    error_bars = [(item[1]['Coefficient'] - item[1]['Confidence Interval'][0], 
                   item[1]['Confidence Interval'][1] - item[1]['Coefficient']) for item in sorted_results]
    colors = [color_dict[item[0]] for item in sorted_results]

    error_bars = np.array(error_bars).T

    # Create plot with adjusted layout
    fig, ax = plt.subplots(figsize=(5, 10))
    fig.subplots_adjust(left=0.3, right=0.9)  # Make space for labels

    # Plot horizontal bars
    bars = ax.barh(
        range(len(labels)), 
        coefficients, 
        xerr=error_bars, 
        color=colors, 
        capsize=5,
        error_kw={'elinewidth': 1, 'capsize': 3, 'ecolor': 'dimgray'}
    )

    # Set y-axis labels with matching colors
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, va='center', ha='right', fontsize=12)
    
    # Color each label
    for i, tick_label in enumerate(ax.get_yticklabels()):
        tick_label.set_color(colors[i])
    
    ax.tick_params(axis='y', which='major', pad=15)

    # Add significance stars
    for i, item in enumerate(sorted_results):
        p_value = item[1]['Corrected P-Value']
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = ''
        
        if significance:
            # Calculate position relative to error bar end
            if coefficients[i] < 0:
                x = coefficients[i] - error_bars[0][i] - 0.0025  # Slightly left of left error bar
            else:
                x = coefficients[i] + error_bars[1][i] + 0.0025  # Slightly right of right error bar
            
            y = i + 0.1  # Keep vertical position slightly above center
            ax.text(x, y, significance, va='center', ha='center', fontsize=12, color='black')

    # Add markers for convergence warnings
    for i, item in enumerate(sorted_results):
        if item[1].get('Convergence Warning', False):
            ax.plot(coefficients[i], i, 'k*', markersize=5, alpha=0.5)

    # Add note about convergence warnings
    n_warnings = sum(1 for item in sorted_results if item[1].get('Convergence Warning', False))
    if n_warnings > 0:
        ax.text(0.98, 0.02, f'* {n_warnings} models had convergence issues', 
                transform=ax.transAxes, ha='right', fontsize=8, style='italic')

    # Final styling
    ax.set_xlabel('Estimated Average Difference \nin Emotion Categories\n(Pre- to Post- 5-MeO-DMT, %)', fontsize=10)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.set_xlim(-0.06, 0.06)
    # xtick size set
    ax.tick_params(axis='x', labelsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    plt.show()
    
    return fig, ax

def export_mlm_results(results_dict, filename=None):
    """
    Export mixed linear model results to a CSV file.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing the MLM results
    filename : str
        Name of the output file
    """
    # Create a list to store results
    results_list = []
    
    # Process each category
    for category, result in results_dict.items():
        ci_lower, ci_upper = result['Confidence Interval']
        
        # Create row
        row = {
            'Category': category,
            'Coefficient': result['Coefficient'],
            'Standard Error': result['Std. Error'],
            'CI Lower': ci_lower,
            'CI Upper': ci_upper,
            'P-Value': result['P-Value'],
            'Corrected P-Value': result['Corrected P-Value'],
            'N Observations': result['Num. Observations'],
            'Convergence Warning': result.get('Convergence Warning', False),
            'Significance': (
                '***' if result['Corrected P-Value'] < 0.001 else
                '**' if result['Corrected P-Value'] < 0.01 else
                '*' if result['Corrected P-Value'] < 0.05 else
                'ns'
            )
        }
        results_list.append(row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Sort by coefficient value
    results_df = results_df.sort_values('Coefficient', ascending=False)
    
    # Round numeric columns
    numeric_cols = ['Coefficient', 'Standard Error', 'CI Lower', 'CI Upper', 'P-Value', 'Corrected P-Value']
    results_df[numeric_cols] = results_df[numeric_cols].round(4)
    
    # Save to CSV if filename provided
    if filename:
        results_df.to_csv(filename, index=False)
    
    return results_df

def save_significant_categories(results_dict, filename='config/mlm_sig_cats.txt'):
    """
    Save categories that showed significant changes (p < 0.05 after correction) to a text file.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing the MLM results
    filename : str
        Path to output file (default: 'config/mlm_sig_cats.txt')
    """
    # Get significant categories
    sig_cats = []
    for category, results in results_dict.items():
        if results['Corrected P-Value'] < 0.05:
            # Store category name without 'liwc_' prefix
            sig_cats.append(category)
    
    # Sort alphabetically
    sig_cats.sort()
    
    # Save to file
    with open(filename, 'w') as f:
        for cat in sig_cats:
            f.write(f"{cat}\n")
    
    print(f"Saved {len(sig_cats)} significant categories to {filename}")
    return sig_cats

if __name__ == '__main__':

    CORE_DIR = '/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals'

    DF = pd.read_csv(f'{CORE_DIR}/data/final/sentence_level.csv')
    print(f'Column count: {len(DF.columns)}')
    DF, GOEMO_COLS = filter_df(DF)
    print(f'Column count after filtering for GoEmotions columns with mean > 1: {len(DF.columns)}')

    # Run mixed effect models for categories from label_dict
    RESULTS_DICT, P_VALUES = run_mixed_effect_models(DF)

    # # Apply p-value correction across all categories
    RESULTS_DICT = apply_p_value_correction(RESULTS_DICT, P_VALUES)

    print(RESULTS_DICT)

    # After getting results_dict
    fig, ax = plot_mixed_effects_results(RESULTS_DICT, f"{CORE_DIR}/outputs/figures/mixed_effects_results_goemo.png")

    # Export results to CSV
    results_df = export_mlm_results(
        RESULTS_DICT,  
        f"{CORE_DIR}/outputs/tables/mlm_results_goemo.csv"
    )
    print("Results exported to CSV")

    # Save significant categories
    sig_cats = save_significant_categories(
        RESULTS_DICT, 
        f"{CORE_DIR}/config/mlm_sig_cats_goemo.txt"
    )


