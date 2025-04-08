import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
import json

def filter_df(df):
    # Get means of numeric columns only
    means = df.select_dtypes(include=['float64', 'int64']).mean()
    
    # Get list of LIWC columns with mean > 1
    liwc_cols = [col for col in df.columns if col.startswith('liwc_') and col in means.index and means[col] > 1]
    
    # Keep non-LIWC columns and LIWC columns with mean > 1
    non_liwc_cols = [col for col in df.columns if not col.startswith('liwc_')]
    df = df[non_liwc_cols + liwc_cols]

    # Remove participant with Participant ID = PRO71
    df = df[df['UserID'] != 'PRO71']

    # only keep columns that are in -14, 15 range for RelativeDate
    df = df[(df['RelativeDate'] >= -14) & (df['RelativeDate'] <= 14)]
    
    return df, liwc_cols

def run_mixed_effect_models(df, label_dict):
    results_dict = {}
    p_values = []

    # Flatten the label_dict to get all category names
    all_categories = []
    for main_cat, sub_cats in label_dict.items():
        all_categories.extend(sub_cats.keys())

    # Reset index to ensure proper alignment
    df = df.reset_index(drop=True)
    
    for category in all_categories:
        # Convert category to its LIWC column name
        liwc_col = f"liwc_{category}"
        
        if liwc_col not in df.columns:
            print(f"Warning: {category} not found in dataframe, skipping...")
            continue
            
        try:
            # Create a clean subset of data for this category
            subset = df[[liwc_col, 'PrePost', 'UserID']].copy()
            subset = subset.dropna()
            subset = subset.reset_index(drop=True)
            
            if len(subset) == 0:
                print(f"Warning: No valid data for {category}, skipping...")
                continue
                
            model = smf.mixedlm(f"{liwc_col} ~ PrePost", subset, groups=subset['UserID'])
            result = model.fit(method='nm', maxiter=1000)
            print(result.summary())
            
            coef = result.params["PrePost"]
            p_value = result.pvalues["PrePost"]
            ci_lower, ci_upper = result.conf_int().loc["PrePost"]
            
            p_values.append(p_value)
            
            # Store results using the original category name (without liwc_ prefix)
            results_dict[category] = {
                "Coefficient": coef,
                "P-Value": p_value,
                "Confidence Interval": (ci_lower, ci_upper),
                "Std. Error": result.bse["PrePost"],
                "Num. Observations": result.nobs
            }
            
            print(f"Successfully processed {category} with {result.nobs} observations")
            
        except Exception as e:
            print(f"Error processing {category}: {str(e)}")
            continue

    return results_dict, p_values

def apply_p_value_correction(results_dict, p_values, alpha=0.05):
    corrected_p_values = smm.multipletests(p_values, alpha=alpha, method='fdr_bh')[1]

    # Add corrected p-values to the results dictionary
    for i, category in enumerate(results_dict.keys()):
        results_dict[category]["Corrected P-Value"] = corrected_p_values[i]

    return results_dict

def plot_mixed_effects_results(results_dict, label_dict, filename='mixed_effects_results.png'):
    # Define a color map for each main category
    category_colors = {
        'Drives': '#8B0000',       # Darker Blue
        'Cognition': '#cc6666',    # Darker Light Red 
        'Affect': '#228b22',       # Darker Green
        'Social': '#1a5a89',       # Darker Red
        'Lifestyle': '#734b9a',    # Darker Purple
        'Physical': '#5c4033',     # Darker Brown
        'Motives': '#c71585',      # Darker Pink
        'Perception': '#595959',   # Darker Gray
        'Conversation': '#8a9a0e', # Darker Yellow-Green
        'Time': '#008b8b',         # Darker Cyan
        'Pronoun': '#e66900',      # Darker Orange
        'Linguistic': '#668b5e'    # Darker Light Green
    }

    # Combine and sort results
    combined_results = {}
    for category, subcats in label_dict.items():
        for subcat in subcats:
            if subcat in results_dict:
                combined_results[subcat] = {
                    'Coefficient': results_dict[subcat]['Coefficient'],
                    'Confidence Interval': results_dict[subcat]['Confidence Interval'],
                    'Corrected P-Value': results_dict[subcat]['Corrected P-Value'],
                    'Category': category
                }

    sorted_results = sorted(
        combined_results.items(),
        key=lambda x: (x[1]['Category'], 'Total' not in x[0], x[0]),
        reverse=True
    )

    # Prepare plotting data
    labels = [label_dict[item[1]['Category']].get(item[0], item[0]) for item in sorted_results]
    coefficients = [item[1]['Coefficient'] for item in sorted_results]
    error_bars = [(item[1]['Coefficient'] - item[1]['Confidence Interval'][0], 
                   item[1]['Confidence Interval'][1] - item[1]['Coefficient']) for item in sorted_results]
    colors = [category_colors[item[1]['Category']] for item in sorted_results]

    error_bars = np.array(error_bars).T

    # Create plot with adjusted layout
    fig, ax = plt.subplots(figsize=(6, 10))
    fig.subplots_adjust(left=0.5, right=0.8)  # Make space for category labels

    # Plot horizontal bars
    bars = ax.barh(
        range(len(labels)), 
        coefficients, 
        xerr=error_bars, 
        color=colors, 
        capsize=5,
        error_kw={'elinewidth': 1, 'capsize': 3, 'ecolor': 'dimgray'}
    )

    # Set y-axis labels (subcategories) with matching colors
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, va='center', ha='right', fontsize=10)
    
    # Color each label according to its category
    for i, tick_label in enumerate(ax.get_yticklabels()):
        category = sorted_results[i][1]['Category']  # Get the category for this label
        tick_label.set_color(category_colors[category])  # Set the label color
    
    ax.tick_params(axis='y', which='major', pad=15)  # Add padding for brackets

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
            # Calculate position based on error bar ends
            if coefficients[i] < 0:
                # For negative coefficients, place near the left error bar end
                x = coefficients[i] - error_bars[0][i]
            else:
                # For positive coefficients, place near the right error bar end
                x = coefficients[i] + error_bars[1][i]
            
            # move a little up
            y = i + 0.2
            ax.text(x, y, significance, va='center', ha='center', fontsize=12, color='black')

    # Main category grouping
    current_category = None
    group_start = 0
    bracket_pairs = []

    # Identify category groups
    for i, item in enumerate(sorted_results):
        category = item[1]['Category']
        if category != current_category:
            if current_category is not None:
                bracket_pairs.append((group_start, i-1, current_category))
            current_category = category
            group_start = i
    bracket_pairs.append((group_start, len(sorted_results)-1, current_category))

    # Draw brackets and category labels
    for start, end, category in bracket_pairs:
        # Vertical positioning
        y_pos = (start + end) / 2

        # Define an x position **outside** the plot (relative to the axis)
        x_pos = -0.6  # Negative values move left, fine-tune as needed

        # Use `ax.transAxes` to place it in figure-relative space
        ax.plot([x_pos, x_pos], [start-0.2, end+0.2], color=category_colors[category], 
                linewidth=2, transform=ax.get_yaxis_transform(), clip_on=False)
        
        # Horizontal lines for the bracket
        ax.plot([x_pos, x_pos+0.02], [start-0.2, start-0.2], color=category_colors[category],
                linewidth=2, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([x_pos, x_pos+0.02], [end+0.2, end+0.2], color=category_colors[category],
                linewidth=2, transform=ax.get_yaxis_transform(), clip_on=False)

        # Add category label, also outside
        ax.text(x_pos-0.05, y_pos, category, color=category_colors[category],
                ha='right', va='center', fontsize=12, fontweight='bold',
                transform=ax.get_yaxis_transform(), clip_on=False)

    # Final styling
    ax.set_xlabel('Estimated Average Difference \nin Vocabulary Categories\n(Pre- to Post- 5-MeO-DMT, %)', fontsize=10)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.set_xlim(-3.2, 3.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    plt.show()
    
    return fig, ax

def export_mlm_results(results_dict, label_dict, filename='mlm_results.csv'):
    """
    Export mixed linear model results to a CSV file.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing the MLM results
    label_dict : dict
        Dictionary containing the category labels
    filename : str
        Name of the output file (default: 'mlm_results.csv')
    """
    # Create a list to store results
    results_list = []
    
    # Process each category
    for main_cat, subcats in label_dict.items():
        for subcat in subcats:
            if subcat in results_dict:
                result = results_dict[subcat]
                ci_lower, ci_upper = result['Confidence Interval']
                
                # Create row
                row = {
                    'Main Category': main_cat,
                    'Subcategory': label_dict[main_cat].get(subcat, subcat),
                    'Raw Category': subcat,
                    'Coefficient': result['Coefficient'],
                    'Standard Error': result['Std. Error'],
                    'CI Lower': ci_lower,
                    'CI Upper': ci_upper,
                    'P-Value': result['P-Value'],
                    'Corrected P-Value': result['Corrected P-Value'],
                    'N Observations': result['Num. Observations'],
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
    
    # Sort by Main Category and Subcategory
    results_df = results_df.sort_values(['Main Category', 'Raw Category'])
    
    # Round numeric columns
    numeric_cols = ['Coefficient', 'Standard Error', 'CI Lower', 'CI Upper', 'P-Value', 'Corrected P-Value']
    results_df[numeric_cols] = results_df[numeric_cols].round(4)
    
    # Save to CSV
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
    # load liwc_labels.json
    with open(f'{CORE_DIR}/config/liwc_labels.json', 'r') as f:
        LABEL_DICT = json.load(f)

    DF = pd.read_csv(f'{CORE_DIR}/data/final/sentence_level.csv')
    print(f'Column count: {len(DF.columns)}')
    DF, LIWC_COLS = filter_df(DF)
    print(f'Column count after filtering for LIWC columns with mean > 1: {len(DF.columns)}')

    # Run mixed effect models for categories from label_dict
    RESULTS_DICT, P_VALUES = run_mixed_effect_models(DF, LABEL_DICT)

    # # Apply p-value correction across all categories
    RESULTS_DICT = apply_p_value_correction(RESULTS_DICT, P_VALUES)

    print(RESULTS_DICT)

    # After getting results_dict
    fig, ax = plot_mixed_effects_results(RESULTS_DICT, LABEL_DICT, f"{CORE_DIR}/outputs/figures/mixed_effects_liwc_results.png")

    # Export results to CSV
    results_df = export_mlm_results(
        RESULTS_DICT, 
        LABEL_DICT, 
        f"{CORE_DIR}/outputs/tables/mlm_results.csv"
    )
    print("Results exported to CSV")

    # Save significant categories
    sig_cats = save_significant_categories(
        RESULTS_DICT, 
        f"{CORE_DIR}/config/mlm_sig_cats.txt"
    )


