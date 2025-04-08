# %%

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_rel, mannwhitneyu
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.utils import resample
import warnings
import re
from collections import defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_rel, mannwhitneyu
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.utils import resample
import warnings
import re
from collections import defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def find_columns(df, pattern):
    """
    Find columns in DataFrame that contain a specific pattern in their names.
    
    Args:
    df (pd.DataFrame): DataFrame to search through.
    pattern (str): Substring pattern to search for in column names.
    
    Returns:
    list: List of column names that match the pattern.
    """
    return [col for col in df.columns if pattern in col]


def calculate_correlation(df, col1, col2):
    """
    Calculate Pearson correlation and p-value between two columns.
    
    Args:
    df (pd.DataFrame): DataFrame containing the data.
    col1 (str): Name of the first column.
    col2 (str): Name of the second column.
    
    Returns:
    tuple: Pearson correlation coefficient and p-value.
    """
    corr, p_value = pearsonr(df[col1], df[col2])
    return corr, p_value


def build_correlation_matrix(df, col_list1, col_list2):
    """
    Build correlation and p-value matrices for two sets of columns.
    
    Args:
    df (pd.DataFrame): DataFrame containing the data.
    col_list1 (list): List of columns for the first group.
    col_list2 (list): List of columns for the second group.
    
    Returns:
    tuple: Correlation matrix and p-value matrix as DataFrames.
    """
    dfx = pd.DataFrame(index=col_list2, columns=col_list1)
    p_values_df = pd.DataFrame(index=col_list2, columns=col_list1)

    for col1 in col_list1:
        for col2 in col_list2:
            corr, p_value = calculate_correlation(df, col1, col2)
            dfx.loc[col2, col1] = corr
            p_values_df.loc[col2, col1] = p_value

    # Convert the DataFrames to float type
    return dfx.astype(float), p_values_df.astype(float)


def filter_and_correct(df_corr, df_pvals, top_n=10, alpha=0.05):
    """
    Filter the top correlations and correct p-values using FDR.
    
    Args:
    df_corr (pd.DataFrame): Correlation matrix.
    df_pvals (pd.DataFrame): P-value matrix.
    top_n (int): Number of top correlations to select.
    alpha (float): Significance level for p-value correction.
    
    Returns:
    pd.DataFrame: DataFrame containing top correlations with corrected p-values.
    """
    combined_df = pd.DataFrame({
        'Correlation': df_corr,
        'P-value': df_pvals
    })

    # Sort by correlation and select the top N
    top_combined_df = combined_df.sort_values(by='Correlation', ascending=False).head(top_n)

    # Perform FDR correction on the p-values
    p_values = top_combined_df['P-value']
    _, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=alpha)
    top_combined_df['Corrected P-value'] = pvals_corrected

    return top_combined_df


def analyze_significant_correlations(df_corr, df_pvals, alpha=0.05):
    """
    Analyze and print significant correlations and corrected p-values.
    
    Args:
    df_corr (pd.DataFrame): Correlation matrix.
    df_pvals (pd.DataFrame): P-value matrix.
    alpha (float): Significance threshold for p-values.
    
    Returns:
    list: List of columns with significant correlations.
    """
    sig_cols = []

    for asc_score in df_corr.columns:
        # Filter correlations and p-values
        filtered_corr = df_corr[asc_score]
        filtered_corr_p = df_pvals[asc_score]

        # Get top correlations and apply FDR correction
        top_combined_df = filter_and_correct(filtered_corr, filtered_corr_p)

        # Print and collect significant correlations
        for index, row in top_combined_df.iterrows():
            if row['P-value'] < alpha:
                print(f"# {asc_score}")
                print(f"{index}, {row['Correlation']}, {row['P-value']}, {row['Corrected P-value']}")
                sig_cols.append(index)

    return sig_cols


def calculate_correlations_and_significant_columns(df, liwc_cols, pattern='survey_ASC_OBN', top_n=10, alpha=0.05):
    """
    Main function to calculate correlations and find significant columns.
    
    Args:
    df (pd.DataFrame): DataFrame containing the data.
    liwc_cols (list): List of columns for LIWC data.
    pattern (str): Substring pattern to match for the other columns.
    top_n (int): Number of top correlations to select.
    alpha (float): Significance threshold for p-values.
    
    Returns:
    tuple: Correlation matrix, p-value matrix, and list of significant columns.
    """
    # Find columns matching the pattern (e.g., 'ASC_OBN')
    quest_test_cols = find_columns(df, pattern)
    
    # Build correlation and p-value matrices
    df_corr, df_pvals = build_correlation_matrix(df, quest_test_cols, liwc_cols)
    
    # Analyze significant correlations and correct p-values
    sig_cols = analyze_significant_correlations(df_corr, df_pvals, alpha=alpha)
    
    return df_corr, df_pvals, sig_cols

def process_html_file(file_path, color_code='#4c82a3ff'):
    """
    Process an HTML file to extract words from <span> tags with a specific color style.

    Args:
    file_path (str): Path to the HTML file.
    color_code (str): Hex color code to search for in the <span> tags (default is '#4c82a3ff').

    Returns:
    dict: Word count dictionary sorted by frequency.
    """
    # Step 1: Read the content of the HTML file into a string
    with open(file_path, 'r') as f:
        html = f.read()

    # Step 2: Convert HTML to lowercase for consistent matching
    html = html.lower()

    # Step 3: Regex to match: <span style='color: #4c82a3ff'>word</span>
    pattern = rf"<span style='color: {color_code}'>(.*?)</span>"

    # Step 4: Extract all matched words
    matches = re.findall(pattern, html)

    # Step 5: Count the occurrences of each word
    word_count = defaultdict(int)
    for word in matches:
        word_count[word] += 1

    # Step 6: Sort the dictionary by count in descending order
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    # Step 7: Convert the sorted list of tuples back into a dictionary
    sorted_word_count_dict = dict(sorted_word_count)

    # discard spaces in dict keys
    sorted_word_count_dict = {k.replace(' ', ''): v for k, v in sorted_word_count_dict.items()}

    return sorted_word_count_dict


def generate_wordcloud(word_count, ax):
    """
    Generate and display a word cloud on a specific axis from a word count dictionary.

    Args:
    word_count (dict): Dictionary of word frequencies.
    ax (matplotlib.axes.Axes): The matplotlib axis on which to draw the word cloud.
    """
    # Step 1: Generate the word cloud using the word count dictionary
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_count)

    # Step 2: Display the word cloud on the provided axis
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')  # No axes for word cloud visualization


CORE_DIR = "/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals"
df = pd.read_csv (f"{CORE_DIR}/data/final/means_level.csv") # per sentence weighed mean
# df.drop(columns=['name'], inplace=True) 
print(len(df))

# %%
try: 
    display(df.head(2))
except: 
    print(df.head(2))

# %%
# sig_cats = open(f"{CORE_DIR}/config/mlm_sig_cats.txt", 'r').read().splitlines()
# liwc_cols = [f"liwc_{cat}_Diff" for cat in sig_cats]
# # only keep liwc_cols in df
# liwc_cols = [col for col in liwc_cols if col in df.columns]
# liwc_cols = ['liwc_Cognition_Diff', 'liwc_Conversation_Diff', 'liwc_Social_Diff']
liwc_cols = [col for col in df.columns if 'liwc' in col and 'Pre' in col]

# only keep rows which don't have NaN in liwc_cols but don't drop other cols
df = df[liwc_cols + ['survey_ASC_OBN']].dropna()

print(liwc_cols)
df_corr, df_pvals, sig_cols = calculate_correlations_and_significant_columns(df, liwc_cols, pattern='survey_ASC_OBN')
print(sig_cols)
# print(df_corr)

# %%

# # Fit the models
# model1 = smf.ols("ASC_OBN~tone_pos_Pre", data=df).fit()
# model2 = smf.ols("ASC_OBN~wellness_Pre", data=df).fit()
# model3 = smf.ols("ASC_OBN~tone_pos_Pre+wellness_Pre", data=df).fit()
# model4 = smf.ols("ASC_OBN~tone_pos_Pre*wellness_Pre", data=df).fit()



# # Correct p-values using fdr-bh
# pvals = [model1.pvalues['tone_pos_Pre'], model2.pvalues['wellness_Pre'], 
#          model3.pvalues[['tone_pos_Pre', 'wellness_Pre']], 
#          model4.pvalues[['tone_pos_Pre', 'wellness_Pre', 'tone_pos_Pre:wellness_Pre']]]
# corrected_pvals = [multipletests(pval, method='fdr_bh')[1] for pval in pvals]

# # Function to extract model results with both original and corrected p-values
# def extract_results(model, model_name, corrected_p):
#     results = {
#         "Model": model_name,
#         "Formula": model.model.formula,
#         "R²": f"{model.rsquared:.3f}",
#         "Adj. R²": f"{model.rsquared_adj:.3f}",
#         "BIC": f"{model.bic:.3f}",
#         "F-stat p-value": f"{model.f_pvalue:.3f}",
#         "Variables": [],
#         "Coefficients": [],
#         "Original p-values": [],
#         "Corrected p-values": []
#     }
    
#     # Add intercept
#     results["Variables"].append("Intercept")
#     results["Coefficients"].append(f"{model.params['Intercept']:.3f}")
#     results["Original p-values"].append(f"{model.pvalues['Intercept']:.3f}")
#     results["Corrected p-values"].append("N/A")  # No correction for intercept
    
#     # Add other variables
#     var_names = [name for name in model.params.index if name != "Intercept"]
#     for i, var in enumerate(var_names):
#         results["Variables"].append(var)
#         results["Coefficients"].append(f"{model.params[var]:.3f}")
#         results["Original p-values"].append(f"{model.pvalues[var]:.3f}")
#         results["Corrected p-values"].append(f"{corrected_p[i]:.3f}")
    
#     return results

# # Extract results from all models
# results = [
#     extract_results(model1, "Model 1", corrected_pvals[0]),
#     extract_results(model2, "Model 2", corrected_pvals[1]),
#     extract_results(model3, "Model 3", corrected_pvals[2]),
#     extract_results(model4, "Model 4", corrected_pvals[3])
# ]

# # Create a DataFrame with the combined results
# combined_df = pd.DataFrame(results)

# # Display the DataFrame
# combined_df