# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
import os
import re
from collections import defaultdict
from wordcloud import WordCloud

def load_data(sentence_level_path):
    """
    Load the sentence level data with LIWC features for analysis.
    
    Parameters:
    -----------
    sentence_level_path : str
        Path to the sentence level CSV file with LIWC features
        
    Returns:
    --------
    pandas.DataFrame
        Prepared dataframe with cognitive and social words features
    """
    # Load the sentence level data
    df = pd.read_csv(sentence_level_path)
    
    # Only include values where RelativeDate is between -14 and 14
    df = df[(df['RelativeDate'] >= -14) & (df['RelativeDate'] <= 14)].copy()
    
    # Ensure the cognitive words feature (liwc_Cognition) is present
    if 'liwc_Cognition' not in df.columns:
        raise ValueError("Cognitive word feature (liwc_Cognition) not found in the data")
    
    # Ensure the social words feature (liwc_Social) is present
    if 'liwc_Social' not in df.columns:
        raise ValueError("Social word feature (liwc_Social) not found in the data")
    
    return df

def filter_valid_users(df, min_entries=2):
    """
    Filter users who have at least a minimum number of entries in both pre and post periods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with cognitive features
    min_entries : int, optional
        Minimum number of entries required in each period
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe with only valid users
    """
    # Find users who have at least min_entries in PrePost = 0
    pre_users = df[df['PrePost'] == 0]['UserID'].value_counts()
    pre_users = pre_users[pre_users >= min_entries].index.tolist()
    
    # Find users who have at least min_entries in PrePost = 1
    post_users = df[df['PrePost'] == 1]['UserID'].value_counts()
    post_users = post_users[post_users >= min_entries].index.tolist()
    
    # Find users who have at least min_entries in both periods
    valid_users = set(pre_users) & set(post_users)
    
    # Filter the dataframe to only include valid users
    return df[df['UserID'].isin(valid_users)]

def calculate_daily_word_usage(df):
    """
    Calculate average word usage per user per day, normalized as deviations from individual means.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with LIWC features
        
    Returns:
    --------
    tuple
        Tuple of DataFrames with standardized daily word usage (cognitive, social)
    """
    # Calculate mean and standard deviation for each user's cognitive words
    user_cog_stats = df.groupby('UserID')['liwc_Cognition'].agg(['mean', 'std']).reset_index()
    # Rename columns for clarity
    user_cog_stats.columns = ['UserID', 'UserCogMean', 'UserCogStd']
    
    # Calculate mean and standard deviation for each user's social words
    user_social_stats = df.groupby('UserID')['liwc_Social'].agg(['mean', 'std']).reset_index()
    # Rename columns for clarity
    user_social_stats.columns = ['UserID', 'UserSocialMean', 'UserSocialStd']
    
    # Merge user stats back to the original dataframe
    df_with_stats = df.merge(user_cog_stats, on='UserID')
    df_with_stats = df_with_stats.merge(user_social_stats, on='UserID')
    
    # Calculate z-scores (deviations in standard deviations from individual mean)
    # Handle cases where standard deviation is 0 to avoid division by zero
    df_with_stats['CogDeviation'] = np.where(
        df_with_stats['UserCogStd'] > 0,
        (df_with_stats['liwc_Cognition'] - df_with_stats['UserCogMean']) / df_with_stats['UserCogStd'],
        0  # If std is 0, deviation is 0
    )
    
    df_with_stats['SocialDeviation'] = np.where(
        df_with_stats['UserSocialStd'] > 0,
        (df_with_stats['liwc_Social'] - df_with_stats['UserSocialMean']) / df_with_stats['UserSocialStd'],
        0  # If std is 0, deviation is 0
    )
    
    # Group by user and day to get average daily deviations
    daily_cognition = df_with_stats.groupby(['UserID', 'RelativeDate'])['CogDeviation'].mean().reset_index()
    daily_social = df_with_stats.groupby(['UserID', 'RelativeDate'])['SocialDeviation'].mean().reset_index()
    
    return daily_cognition, daily_social

def aggregate_daily_word_usage(daily_usage_df, feature_col):
    """
    Aggregate daily word usage across all users.
    
    Parameters:
    -----------
    daily_usage_df : pandas.DataFrame
        Dataframe with daily word usage values per user
    feature_col : str
        Column name of the feature to analyze
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with aggregated daily word usage across all users
    """
    # Group by day to aggregate across all users
    agg_daily = daily_usage_df.groupby('RelativeDate')[feature_col].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error
    agg_daily['se'] = agg_daily['std'] / np.sqrt(agg_daily['count'])
    
    # Create a dataframe with all days from -14 to 14
    all_days = pd.DataFrame({'RelativeDate': np.arange(-14, 15, 0.5 if 0.5 in daily_usage_df['RelativeDate'].values else 1)})
    
    # Merge with aggregated data, but don't fill NaN values with 0
    agg_daily = pd.merge(all_days, agg_daily, on='RelativeDate', how='left')
    
    # Apply smoothing for visualization only on non-NaN values
    valid_means = ~agg_daily['mean'].isna()
    if valid_means.any():
        # Create a copy of the mean values
        smoothed_values = agg_daily['mean'].copy()
        # Apply smoothing only on the valid values
        valid_indices = agg_daily.index[valid_means]
        valid_values = agg_daily.loc[valid_indices, 'mean']
        if len(valid_values) > 3:  # Only smooth if we have enough data points
            smoothed_values.loc[valid_indices] = gaussian_filter1d(valid_values, sigma=1)
        agg_daily['smoothed_mean'] = smoothed_values
    else:
        agg_daily['smoothed_mean'] = agg_daily['mean']
    
    return agg_daily.sort_values('RelativeDate')

def calculate_period_statistics(daily_usage_df, feature_col, agg_daily_df=None):
    """
    Calculate statistics for different periods around the dosing.
    
    Parameters:
    -----------
    daily_usage_df : pandas.DataFrame
        The dataframe with daily word usage values
    feature_col : str
        Column name of the feature to analyze
    agg_daily_df : pandas.DataFrame, optional
        Dataframe with aggregated data including user counts to filter out days with insufficient data
        
    Returns:
    --------
    dict
        Dictionary with period statistics
    tuple
        Tuple containing the periods dictionary
    """
    # Define periods as weekly ranges
    periods = {
        'Week -2': (-14, -7),
        'Week -1': (-7, 0),
        'Week +1': (0, 7),
        'Week +2': (7, 14)
    }
    
    # Filter out days with only one user if agg_daily_df is provided
    if agg_daily_df is not None:
        # Get days with more than one user
        valid_days = agg_daily_df[agg_daily_df['count'] > 1]['RelativeDate'].values
        # Filter daily_usage_df to only include those days
        daily_usage_df = daily_usage_df[daily_usage_df['RelativeDate'].isin(valid_days)].copy()
    
    # Calculate statistics for each period
    period_stats = {}
    
    for period_name, (start, end) in periods.items():
        period_data = daily_usage_df[
            (daily_usage_df['RelativeDate'] >= start) & 
            (daily_usage_df['RelativeDate'] < end)
        ]
        
        period_stats[period_name] = {
            'mean': period_data[feature_col].mean(),
            'std': period_data[feature_col].std(),
            'count': len(period_data),
            'se': period_data[feature_col].std() / np.sqrt(len(period_data)),
            'days': end - start
        }
    
    return period_stats, periods

def pairwise_period_comparisons(daily_usage_df, periods, feature_col, agg_daily_df=None):
    """
    Perform pairwise t-tests between periods.
    
    Parameters:
    -----------
    daily_usage_df : pandas.DataFrame
        Dataframe with daily word usage values
    periods : dict
        Dictionary with period definitions
    feature_col : str
        Column name of the feature to analyze
    agg_daily_df : pandas.DataFrame, optional
        Dataframe with aggregated data including user counts to filter out days with insufficient data
        
    Returns:
    --------
    list
        List of dictionaries with comparison results
    """
    # Filter out days with only one user if agg_daily_df is provided
    if agg_daily_df is not None:
        # Get days with more than one user
        valid_days = agg_daily_df[agg_daily_df['count'] > 1]['RelativeDate'].values
        # Filter daily_usage_df to only include those days
        daily_usage_df = daily_usage_df[daily_usage_df['RelativeDate'].isin(valid_days)].copy()
    
    # Get all period names
    period_names = list(periods.keys())
    
    # Initialize list to store comparison results
    comparisons = []
    
    # Perform all pairwise comparisons
    for i in range(len(period_names)):
        for j in range(i+1, len(period_names)):
            period1 = period_names[i]
            period2 = period_names[j]
            
            # Get data for period 1
            start1, end1 = periods[period1]
            period1_data = daily_usage_df[
                (daily_usage_df['RelativeDate'] >= start1) & 
                (daily_usage_df['RelativeDate'] < end1)
            ][feature_col]
            
            # Get data for period 2
            start2, end2 = periods[period2]
            period2_data = daily_usage_df[
                (daily_usage_df['RelativeDate'] >= start2) & 
                (daily_usage_df['RelativeDate'] < end2)
            ][feature_col]
            
            # Perform t-test
            t_stat, p_val = ttest_ind(period1_data, period2_data, equal_var=False)
            
            # Store results
            comparisons.append({
                'Period 1': period1,
                'Period 2': period2,
                'mean1': period1_data.mean(),
                'mean2': period2_data.mean(),
                't-statistic': t_stat,
                'p-value': p_val
            })
    
    # Apply FDR correction for multiple comparisons
    p_values = [comp['p-value'] for comp in comparisons]
    rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Add corrected p-values to comparisons
    for i, comp in enumerate(comparisons):
        comp['p-value-fdr'] = p_corrected[i]
    
    return comparisons

def print_detailed_statistics(comparisons, feature_name):
    """
    Print detailed statistical test results in a formatted manner.
    
    Parameters:
    -----------
    comparisons : list
        List of dictionaries with comparison results
    feature_name : str
        Name of the feature being compared
    """
    print(f"\n--- {feature_name.upper()} WORD USAGE COMPARISONS ---\n")
    
    for comp in comparisons:
        period1 = comp['Period 1']
        period2 = comp['Period 2']
        mean1 = comp['mean1']
        mean2 = comp['mean2']
        t_stat = comp['t-statistic']
        p_val = comp['p-value']
        p_fdr = comp['p-value-fdr']
        significance = ''
        if p_fdr < 0.001:
            significance = '***'
        elif p_fdr < 0.01:
            significance = '**'
        elif p_fdr < 0.05:
            significance = '*'
        else:
            significance = 'ns'
        
        print(f"  {period1} vs {period2}:")
        print(f"    Mean: {mean1:.2f} vs {mean2:.2f}")
        print(f"    t = {t_stat:.2f}, p = {p_val:.4f}, p-fdr = {p_fdr:.4f} {significance}")

def create_daily_stats_table(agg_daily_cog, agg_daily_social, output_path=None):
    """
    Create a table with daily means and user counts.
    
    Parameters:
    -----------
    agg_daily_cog : pandas.DataFrame
        Dataframe with aggregated daily cognitive word usage
    agg_daily_social : pandas.DataFrame
        Dataframe with aggregated daily social word usage
    output_path : str, optional
        Path to save the output table
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the daily statistics
    """
    # Merge cognitive and social dataframes on RelativeDate
    daily_stats = pd.merge(
        agg_daily_cog[['RelativeDate', 'mean', 'count']],
        agg_daily_social[['RelativeDate', 'mean', 'count']],
        on='RelativeDate',
        suffixes=('_cog', '_social'),
        how='outer'
    )
    
    # Reorder columns for better readability
    daily_stats = daily_stats[['RelativeDate', 'mean_cog', 'count_cog', 'mean_social', 'count_social']]
    
    # Rename columns for clarity
    daily_stats.columns = ['Day', 'Cognitive Mean', 'Cognitive Users', 'Social Mean', 'Social Users']
    
    # Filter to only include full days and special half days (-0.5 and 0.5)
    daily_stats = daily_stats[
        (daily_stats['Day'].apply(lambda x: x.is_integer())) | 
        (daily_stats['Day'] == -0.5) | 
        (daily_stats['Day'] == 0.5)
    ]
    
    # Sort by day
    daily_stats = daily_stats.sort_values('Day')
    
    # Save to CSV if output path is provided
    if output_path:
        daily_stats.to_csv(f"{output_path}_daily_stats.csv", index=False)
        
        # Also create a formatted text table
        with open(f"{output_path}_daily_stats.txt", 'w') as f:
            f.write("Daily Statistics for Cognitive and Social Word Usage\n")
            f.write("==================================================\n\n")
            f.write(f"{'Day':>5} | {'Cognitive Mean':>15} | {'Cognitive Users':>15} | {'Social Mean':>15} | {'Social Users':>15}\n")
            f.write(f"{'-'*5} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15}\n")
            
            for _, row in daily_stats.iterrows():
                day = row['Day']
                
                # Handle potentially NaN values
                if pd.isna(row['Cognitive Mean']):
                    cog_mean_str = "N/A".rjust(15)
                else:
                    cog_mean_str = f"{row['Cognitive Mean']:15.3f}"
                    
                if pd.isna(row['Cognitive Users']):
                    cog_users_str = "N/A".rjust(15)
                else:
                    cog_users_str = f"{int(row['Cognitive Users']):15d}"
                    
                if pd.isna(row['Social Mean']):
                    social_mean_str = "N/A".rjust(15)
                else:
                    social_mean_str = f"{row['Social Mean']:15.3f}"
                    
                if pd.isna(row['Social Users']):
                    social_users_str = "N/A".rjust(15)
                else:
                    social_users_str = f"{int(row['Social Users']):15d}"
                
                f.write(f"{day:5.1f} | {cog_mean_str} | {cog_users_str} | {social_mean_str} | {social_users_str}\n")
    
    return daily_stats

def create_statistical_results_table(period_stats_cog, period_stats_social, comparisons_cog, comparisons_social, output_path=None):
    """
    Create a comprehensive table of statistical results for supplementary materials.
    
    Parameters:
    -----------
    period_stats_cog : dict
        Dictionary with period statistics for cognitive words
    period_stats_social : dict
        Dictionary with period statistics for social words
    comparisons_cog : list
        List of dictionaries with comparison results for cognitive words
    comparisons_social : list
        List of dictionaries with comparison results for social words
    output_path : str, optional
        Path to save the output table
        
    Returns:
    --------
    tuple
        Tuple containing two DataFrames (period stats and comparisons)
    """
    # Create period statistics DataFrame
    period_stats_data = []
    for period in period_stats_cog.keys():
        period_stats_data.append({
            'Period': period,
            'Cognitive_Mean': period_stats_cog[period]['mean'],
            'Cognitive_SE': period_stats_cog[period]['se'],
            'Cognitive_N': period_stats_cog[period]['count'],
            'Social_Mean': period_stats_social[period]['mean'],
            'Social_SE': period_stats_social[period]['se'],
            'Social_N': period_stats_social[period]['count']
        })
    
    period_stats_df = pd.DataFrame(period_stats_data)
    
    # Create pairwise comparisons DataFrame
    comparison_data = []
    for cog, soc in zip(comparisons_cog, comparisons_social):
        # Add significance notation for cognitive words
        if cog['p-value-fdr'] < 0.001:
            cog_sig = '***'
        elif cog['p-value-fdr'] < 0.01:
            cog_sig = '**'
        elif cog['p-value-fdr'] < 0.05:
            cog_sig = '*'
        else:
            cog_sig = 'ns'
            
        # Add significance notation for social words
        if soc['p-value-fdr'] < 0.001:
            soc_sig = '***'
        elif soc['p-value-fdr'] < 0.01:
            soc_sig = '**'
        elif soc['p-value-fdr'] < 0.05:
            soc_sig = '*'
        else:
            soc_sig = 'ns'
            
        comparison_data.append({
            'Comparison': f"{cog['Period 1']} vs {cog['Period 2']}",
            'Cognitive_t': cog['t-statistic'],
            'Cognitive_p': cog['p-value'],
            'Cognitive_p_FDR': cog['p-value-fdr'],
            'Cognitive_Sig': cog_sig,
            'Cognitive_Mean_Diff': cog['mean1'] - cog['mean2'],
            'Social_t': soc['t-statistic'],
            'Social_p': soc['p-value'],
            'Social_p_FDR': soc['p-value-fdr'],
            'Social_Sig': soc_sig,
            'Social_Mean_Diff': soc['mean1'] - soc['mean2']
        })
    
    comparisons_df = pd.DataFrame(comparison_data)
    
    # Reorder columns to put significance right after FDR p-values
    comparisons_df = comparisons_df[[
        'Comparison',
        'Cognitive_t',
        'Cognitive_p',
        'Cognitive_p_FDR',
        'Cognitive_Sig',
        'Cognitive_Mean_Diff',
        'Social_t',
        'Social_p',
        'Social_p_FDR',
        'Social_Sig',
        'Social_Mean_Diff'
    ]]
    
    # Save as CSV files if output path is provided
    if output_path:
        # Create tables directory if it doesn't exist
        tables_dir = os.path.join(os.path.dirname(output_path), '..', 'tables')
        os.makedirs(tables_dir, exist_ok=True)
        
        # Save tables to the tables directory
        tables_path = os.path.join(tables_dir, os.path.basename(output_path))
        period_stats_df.to_csv(f"{tables_path}_period_statistics.csv", index=False)
        comparisons_df.to_csv(f"{tables_path}_pairwise_comparisons.csv", index=False)
        
        print(f"\nStatistical results tables saved to:")
        print(f"- {tables_path}_period_statistics.csv")
        print(f"- {tables_path}_pairwise_comparisons.csv")
    
    return period_stats_df, comparisons_df

def plot_combined_visualization(agg_daily_cog, agg_daily_social, periods, daily_cognition=None, 
                               daily_social=None, period_stats_cog=None, period_stats_social=None, 
                               comparisons_cog=None, comparisons_social=None, output_path=None):
    """
    Create a single combined visualization with time series, word clouds, and bar charts.
    
    Parameters:
    -----------
    agg_daily_cog : pandas.DataFrame
        Dataframe with aggregated daily cognitive word usage
    agg_daily_social : pandas.DataFrame
        Dataframe with aggregated daily social word usage
    periods : dict
        Dictionary with period definitions
    daily_cognition : pandas.DataFrame, optional
        Dataframe with individual user daily cognitive word usage
    daily_social : pandas.DataFrame, optional
        Dataframe with individual user daily social word usage
    period_stats_cog : dict, optional
        Dictionary with period statistics for cognitive words
    period_stats_social : dict, optional
        Dictionary with period statistics for social words
    comparisons_cog : list, optional
        List of dictionaries with comparison results for cognitive words
    comparisons_social : list, optional
        List of dictionaries with comparison results for social words
    output_path : str, optional
        Path to save the output figure
    
    Returns:
    --------
    tuple
        Tuple containing the figure and axes
    """
    # Create figure with 3 rows and 2 columns with specific height ratios
    fig = plt.figure(figsize=(14, 14))
    
    # Create a gridspec with control over spacing and heights
    gs = fig.add_gridspec(4, 2, height_ratios=[2,2,2, 2], hspace=0.3, wspace=0.3)
    
    # Create axes using the gridspec
    ax_timeseries = fig.add_subplot(gs[:2, :])  # Top row - time series (spans both columns)
    ax_wordcloud_cog = fig.add_subplot(gs[2, 0])  # Middle row, left - cognitive wordcloud
    ax_wordcloud_soc = fig.add_subplot(gs[2, 1])  # Middle row, right - social wordcloud
    ax_bar_cog = fig.add_subplot(gs[3, 0])  # Bottom row, left - cognitive bar chart
    ax_bar_soc = fig.add_subplot(gs[3, 1])  # Bottom row, right - social bar chart
    
    # Define custom colors with slightly darker variants for certain elements
    cog_color = '#1f77b4'  # Rich medium blue for cognitive words
    social_color = '#d62728'  # Deep red for social words
    
    # Filter out days with insufficient data (count <= 1) for line plots only
    agg_daily_cog_filtered = agg_daily_cog[agg_daily_cog['count'] > 1].copy()
    agg_daily_social_filtered = agg_daily_social[agg_daily_social['count'] > 1].copy()
    
    # Split data into pre and post dosing segments to create a break for cognitive words
    # Filter out NaN values to avoid plotting gaps
    pre_dosing_cog = agg_daily_cog_filtered[(agg_daily_cog_filtered['RelativeDate'] <= -0.5) & 
                                           (~agg_daily_cog_filtered['smoothed_mean'].isna())]
    post_dosing_cog = agg_daily_cog_filtered[(agg_daily_cog_filtered['RelativeDate'] >= 0.5) & 
                                            (~agg_daily_cog_filtered['smoothed_mean'].isna())]
    
    # Split data into pre and post dosing segments to create a break for social words
    # Filter out NaN values to avoid plotting gaps
    pre_dosing_social = agg_daily_social_filtered[(agg_daily_social_filtered['RelativeDate'] <= -0.5) & 
                                                 (~agg_daily_social_filtered['smoothed_mean'].isna())]
    post_dosing_social = agg_daily_social_filtered[(agg_daily_social_filtered['RelativeDate'] >= 0.5) & 
                                                  (~agg_daily_social_filtered['smoothed_mean'].isna())]
    
    # Plot all individual data points from all users if provided - with slightly higher opacity
    if daily_cognition is not None:
        # Show all individual data points with increased opacity
        ax_timeseries.scatter(daily_cognition['RelativeDate'], daily_cognition['CogDeviation'], 
                  color=cog_color, alpha=0.3, s=15, label='Individual Cognitive Data Points')
    
    if daily_social is not None:
        # Show all individual data points with increased opacity
        ax_timeseries.scatter(daily_social['RelativeDate'], daily_social['SocialDeviation'], 
                  color=social_color, alpha=0.3, s=15, label='Individual Social Data Points')
    
    # Plot pre-dosing segment for cognitive words with increased line width
    if not pre_dosing_cog.empty:
        ax_timeseries.plot(pre_dosing_cog['RelativeDate'], pre_dosing_cog['smoothed_mean'], 
                color=cog_color, linewidth=2.5, label='Cognitive Word Usage (Smoothed)')
        
        # Add confidence interval for pre-dosing cognitive words with increased opacity
        ax_timeseries.fill_between(
            pre_dosing_cog['RelativeDate'],
            pre_dosing_cog['smoothed_mean'] - pre_dosing_cog['se'],
            pre_dosing_cog['smoothed_mean'] + pre_dosing_cog['se'],
            color=cog_color, alpha=0.3
        )
    
    # Plot post-dosing segment for cognitive words with increased line width
    if not post_dosing_cog.empty:
        ax_timeseries.plot(post_dosing_cog['RelativeDate'], post_dosing_cog['smoothed_mean'], 
                color=cog_color, linewidth=2.5, label=None if not pre_dosing_cog.empty else 'Cognitive Word Usage (Smoothed)')
        
        # Add confidence interval for post-dosing cognitive words with increased opacity
        ax_timeseries.fill_between(
            post_dosing_cog['RelativeDate'],
            post_dosing_cog['smoothed_mean'] - post_dosing_cog['se'],
            post_dosing_cog['smoothed_mean'] + post_dosing_cog['se'],
            color=cog_color, alpha=0.3
        )
    
    # Plot pre-dosing segment for social words with increased line width
    if not pre_dosing_social.empty:
        ax_timeseries.plot(pre_dosing_social['RelativeDate'], pre_dosing_social['smoothed_mean'], 
                color=social_color, linewidth=2.5, label='Social Word Usage (Smoothed)')
        
        # Add confidence interval for pre-dosing social words with increased opacity
        ax_timeseries.fill_between(
            pre_dosing_social['RelativeDate'],
            pre_dosing_social['smoothed_mean'] - pre_dosing_social['se'],
            pre_dosing_social['smoothed_mean'] + pre_dosing_social['se'],
            color=social_color, alpha=0.3
        )
    
    # Plot post-dosing segment for social words with increased line width
    if not post_dosing_social.empty:
        ax_timeseries.plot(post_dosing_social['RelativeDate'], post_dosing_social['smoothed_mean'], 
                color=social_color, linewidth=2.5, label=None if not pre_dosing_social.empty else 'Social Word Usage (Smoothed)')
        
        # Add confidence interval for post-dosing social words with increased opacity
        ax_timeseries.fill_between(
            post_dosing_social['RelativeDate'],
            post_dosing_social['smoothed_mean'] - post_dosing_social['se'],
            post_dosing_social['smoothed_mean'] + post_dosing_social['se'],
            color=social_color, alpha=0.3
        )
    
    # Add vertical line at day 0 (dosing)
    ax_timeseries.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Dosing Day')
    
    # Replace colored background with vertical lines and text labels for periods
    period_edges = sorted(set(edge for period, (start, end) in periods.items() for edge in [start, end]))
    
    # Add vertical lines at period boundaries
    for edge in period_edges:
        if edge != 0:  # We already have a line at day 0
            ax_timeseries.axvline(x=edge, color='gray', linestyle=':', alpha=0.5)
    
    # Add text labels for periods in the top figure
    y_text_pos = ax_timeseries.get_ylim()[1] * 0.9  # Position text near the top
    full_period_labels = {
        'Week -2': 'Week -2 (-14 to -7)',
        'Week -1': 'Week -1 (-7 to 0)',
        'Week +1': 'Week +1 (0 to 7)',
        'Week +2': 'Week +2 (7 to 14)'
    }
    for period_name, (start, end) in periods.items():
        # Use the full label for the top figure
        label = full_period_labels[period_name]
        # Place text in the middle of the period
        mid_point = (start + end) / 2
        ax_timeseries.text(mid_point, y_text_pos, label, 
                          ha='center', va='bottom', fontsize=9,
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # Add labels and title
    ax_timeseries.set_xlabel('Days Relative to Dosing', fontsize=12)
    ax_timeseries.set_ylabel('Word Usage \n(Standard Deviations from Individual Mean)', fontsize=12)
    ax_timeseries.set_title('Temporal Evolution of Cognitive and Social Word Usage Around 5-MeO-DMT Dosing', fontsize=14)
    
    # Add horizontal line at y=0 to represent individual baseline
    ax_timeseries.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add grid
    ax_timeseries.grid(True, alpha=0.3)
    
    # Add legend - place it inside the plot instead of outside
    ax_timeseries.legend(loc='best', fontsize=9)
    
    # Set x-axis limits to focus on the period of interest
    ax_timeseries.set_xlim(-14.5, 14.5)
    
    # --- WORD CLOUDS (MIDDLE ROW) ---
    
    CORE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        # Process wordcloud data
        def process_html_file(file_path, color_code='#4c82a3ff'):
            """Process an HTML file to extract words from <span> tags with a specific color style."""
            import re
            from collections import defaultdict
            
            # Read the content of the HTML file into a string
            with open(file_path, 'r') as f:
                html = f.read()

            # Convert HTML to lowercase for consistent matching
            html = html.lower()

            # Regex to match: <span style='color: #4c82a3ff'>word</span>
            pattern = rf"<span style='color: {color_code}'>(.*?)</span>"

            # Extract all matched words
            matches = re.findall(pattern, html)

            # Count the occurrences of each word
            word_count = defaultdict(int)
            for word in matches:
                word_count[word] += 1

            # Sort the dictionary by count in descending order
            sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

            # Convert the sorted list of tuples back into a dictionary
            sorted_word_count_dict = dict(sorted_word_count)

            # discard spaces in dict keys
            sorted_word_count_dict = {k.replace(' ', ''): v for k, v in sorted_word_count_dict.items()}

            return sorted_word_count_dict
        
        # Try to load the data files
        try:
            wcd_cog = process_html_file(os.path.join(CORE_DIR, 'data/wordcloud_data/cognition.html'))
            wcd_soc = process_html_file(os.path.join(CORE_DIR, 'data/wordcloud_data/social.html'))
            print("COGNITION: ", wcd_cog)
            print("SOCIAL: ", wcd_soc)

            # Cognitive wordcloud - update colormap to a darker blue variant
            wordcloud_cog = WordCloud(width=900, height=400, background_color='white', 
                                      max_words=75, colormap='Blues_r', random_state=10).generate_from_frequencies(wcd_cog)
            ax_wordcloud_cog.imshow(wordcloud_cog, interpolation='bilinear')
            ax_wordcloud_cog.axis('off')
            # ax_wordcloud_cog.set_title("Comparison of Cognitive Word Usage", fontsize=14)
            
            # Social wordcloud - update colormap to a darker red variant
            wordcloud_soc = WordCloud(width=900, height=400, background_color='white', 
                                      max_words=75, colormap='Reds_r', random_state=42).generate_from_frequencies(wcd_soc)
            ax_wordcloud_soc.imshow(wordcloud_soc, interpolation='bilinear')
            ax_wordcloud_soc.axis('off')
            # ax_wordcloud_soc.set_title("Comparison of Social Word Usage", fontsize=14)
            
        except FileNotFoundError as e:
            print(f"Warning: Could not find wordcloud data files: {e}")
            ax_wordcloud_cog.text(0.5, 0.5, "Wordcloud data not available", ha='center', va='center')
            ax_wordcloud_cog.axis('off')
            ax_wordcloud_soc.text(0.5, 0.5, "Wordcloud data not available", ha='center', va='center')
            ax_wordcloud_soc.axis('off')
            
    except ImportError:
        print("Warning: WordCloud package not installed. Wordclouds will not be displayed.")
        ax_wordcloud_cog.text(0.5, 0.5, "WordCloud package not installed", ha='center', va='center')
        ax_wordcloud_cog.axis('off')
        ax_wordcloud_soc.text(0.5, 0.5, "WordCloud package not installed", ha='center', va='center')
        ax_wordcloud_soc.axis('off')
    
    # --- BAR PLOTS (BOTTOM ROW) ---
    
    if period_stats_cog and period_stats_social and comparisons_cog and comparisons_social:
        # Plot for cognitive words (bottom left) with increased opacity
        periods_list = list(period_stats_cog.keys())
        means_cog = [stats['mean'] for stats in period_stats_cog.values()]
        errors_cog = [stats['se'] for stats in period_stats_cog.values()]
        
        bars_cog = ax_bar_cog.bar(range(len(periods_list)), means_cog, yerr=errors_cog, capsize=5, 
                       color=cog_color, alpha=0.8, edgecolor=cog_color)
        
        # Add significance bars for cognitive word comparisons
        min_y = min(means_cog) - max(errors_cog) * 2
        max_y = max(means_cog) + max(errors_cog) * 2
        
        # If all values have the same sign, ensure zero is included
        if min_y > 0:
            min_y = 0
        if max_y < 0:
            max_y = 0
            
        bar_height = (max_y - min_y) * 0.05
        
        # Start significance bars from the top of the highest bar
        y_pos = max_y + bar_height
        
        for comp in comparisons_cog:
            if comp['p-value-fdr'] < 0.05:  # Only show significant comparisons
                idx1 = periods_list.index(comp['Period 1'])
                idx2 = periods_list.index(comp['Period 2'])
                
                # Draw the line
                ax_bar_cog.plot([idx1, idx2], [y_pos, y_pos], color='black', linewidth=1.5)
                
                # Add significance stars
                if comp['p-value-fdr'] < 0.001:
                    stars = '***'
                elif comp['p-value-fdr'] < 0.01:
                    stars = '**'
                else:
                    stars = '*'
                    
                ax_bar_cog.text((idx1 + idx2)/2, y_pos - 0.005, stars, 
                        ha='center', va='bottom', color='black')
                
                y_pos += bar_height * 1.5  # Increment for next bar
        
        # Add horizontal line at y=0
        ax_bar_cog.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Set y-axis limit to accommodate significance bars and include zero
        buffer = (y_pos - max_y) * 0.2  # Add a small buffer
        # Limit how high the y-axis can go
        max_ylim = max(means_cog) + max(errors_cog) * 4  # Set a reasonable maximum height
        y_pos_capped = min(y_pos + buffer, max_ylim)
        ax_bar_cog.set_ylim(min_y - buffer, y_pos_capped)
        
        # Add labels for cognitive words
        ax_bar_cog.set_xlabel('Time Period Relative to Dosing', fontsize=12)
        ax_bar_cog.set_ylabel('Mean Cognitive Word Usage\n(Standard Deviations \nfrom Individual Mean)', 
                          fontsize=12, labelpad=15)
        
        # Set x-tick labels
        ax_bar_cog.set_xticks(range(len(periods_list)))
        ax_bar_cog.set_xticklabels(periods_list, rotation=45, ha='right', fontsize=10)
        
        # Add grid
        ax_bar_cog.grid(True, alpha=0.3, axis='y')
        ax_bar_cog.set_title("Cognitive Word Usage", fontsize=14)
        
        # Plot for social words (bottom right) with increased opacity
        means_social = [stats['mean'] for stats in period_stats_social.values()]
        errors_social = [stats['se'] for stats in period_stats_social.values()]
        
        bars_soc = ax_bar_soc.bar(range(len(periods_list)), means_social, yerr=errors_social, capsize=5, 
                       color=social_color, alpha=0.8, edgecolor=social_color)
        
        # Add significance bars for social word comparisons
        min_y = min(means_social) - max(errors_social) * 2
        max_y = max(means_social) + max(errors_social) * 2
        
        # If all values have the same sign, ensure zero is included
        if min_y > 0:
            min_y = 0
        if max_y < 0:
            max_y = 0
            
        bar_height = (max_y - min_y) * 0.05
        
        # Start significance bars from the top of the highest bar
        y_pos = max_y + bar_height
        
        for comp in comparisons_social:
            if comp['p-value-fdr'] < 0.05:  # Only show significant comparisons
                idx1 = periods_list.index(comp['Period 1'])
                idx2 = periods_list.index(comp['Period 2'])
                
                # Draw the line
                ax_bar_soc.plot([idx1, idx2], [y_pos, y_pos], color='black', linewidth=1.5)
                
                # Add significance stars
                if comp['p-value-fdr'] < 0.001:
                    stars = '***'
                elif comp['p-value-fdr'] < 0.01:
                    stars = '**'
                else:
                    stars = '*'
                    
                ax_bar_soc.text((idx1 + idx2)/2, y_pos + bar_height*0.1, stars, 
                        ha='center', va='bottom', color='black')
                
                y_pos += bar_height * 1.5  # Increment for next bar
        
        # Add horizontal line at y=0
        ax_bar_soc.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Set y-axis limit to accommodate significance bars and include zero
        buffer = (y_pos - max_y) * 0.2  # Add a small buffer
        # Limit how high the y-axis can go
        max_ylim = max(means_social) + max(errors_social) * 4  # Set a reasonable maximum height
        y_pos_capped = min(y_pos + buffer, max_ylim)
        ax_bar_soc.set_ylim(min_y - buffer, y_pos_capped)
        
        # Add labels for social words
        ax_bar_soc.set_xlabel('Time Period Relative to Dosing', fontsize=12)
        ax_bar_soc.set_ylabel('Mean Social Word Usage\n(Standard Deviations \nfrom Individual Mean)', 
                          fontsize=12, labelpad=15)
        
        # Set x-tick labels
        ax_bar_soc.set_xticks(range(len(periods_list)))
        ax_bar_soc.set_xticklabels(periods_list, rotation=45, ha='right', fontsize=10)
        
        # Add grid
        ax_bar_soc.grid(True, alpha=0.3, axis='y')
        ax_bar_soc.set_title("Social Word Usage", fontsize=14)

    else:
        # Show message if period stats not provided
        ax_bar_cog.text(0.5, 0.5, "Period statistics not available", ha='center', va='center')
        ax_bar_cog.axis('off')
        ax_bar_soc.text(0.5, 0.5, "Period statistics not available", ha='center', va='center')
        ax_bar_soc.axis('off')
    
    # Adjust the overall layout
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Combined visualization saved to: {output_path}")
    
    return fig, (ax_timeseries, ax_wordcloud_cog, ax_wordcloud_soc, ax_bar_cog, ax_bar_soc)

def add_period_column(df):
    """
    Add a period column to the dataframe based on the relative date.
    0: -14 to -7
    1: -7 to 0
    2: 0 to 7
    3: 7 to 14
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with RelativeDate column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added Period column
    """
    df['Period'] = pd.cut(
        df['RelativeDate'], 
        bins=[-15, -7, 0, 7, 15], 
        labels=[0, 1, 2, 3],
        include_lowest=True
    )
    return df

def analyze_word_dynamics(sentence_level_path, output_path=None):
    """
    Perform complete cognitive and social word dynamics analysis.
    
    Parameters:
    -----------
    sentence_level_path : str
        Path to the sentence level CSV file with LIWC features
    output_path : str, optional
        Path to save the output figure
        
    Returns:
    --------
    tuple
        Tuple containing the dataframe with word usage data and analysis results
    """
    print("\n=== RUNNING COGNITIVE AND SOCIAL WORD DYNAMICS ANALYSIS ===\n")
    
    # Load data
    df = load_data(sentence_level_path)
    print(f"Data loaded successfully. {len(df)} entries found.")
    
    # Filter valid users
    df = filter_valid_users(df)
    print(f"Filtered valid users: {df['UserID'].nunique()} users remaining.")
    
    # Add period column
    df = add_period_column(df)
    print("Period column added.")
    
    # Calculate daily word usage (as standardized deviations from individual means)
    daily_cognition, daily_social = calculate_daily_word_usage(df)
    print("Daily word usage calculated as deviations from individual means.")
    
    # Aggregate daily word usage
    agg_daily_cog = aggregate_daily_word_usage(daily_cognition, 'CogDeviation')
    agg_daily_social = aggregate_daily_word_usage(daily_social, 'SocialDeviation')
    print("Daily word usage deviations aggregated across users.")
    
    # Calculate period statistics for cognitive words - now passing agg_daily_cog to filter by user count
    period_stats_cog, periods = calculate_period_statistics(
        daily_cognition, 'CogDeviation', agg_daily_df=agg_daily_cog
    )
    
    # Calculate period statistics for social words - now passing agg_daily_social to filter by user count
    period_stats_social, _ = calculate_period_statistics(
        daily_social, 'SocialDeviation', agg_daily_df=agg_daily_social
    )
    print("Period statistics calculated (excluding days with only one user).")
    
    # Perform pairwise comparisons for cognitive words - now passing agg_daily_cog to filter by user count
    comparisons_cog = pairwise_period_comparisons(
        daily_cognition, periods, 'CogDeviation', agg_daily_df=agg_daily_cog
    )
    
    # Perform pairwise comparisons for social words - now passing agg_daily_social to filter by user count
    comparisons_social = pairwise_period_comparisons(
        daily_social, periods, 'SocialDeviation', agg_daily_df=agg_daily_social
    )
    print("Pairwise comparisons completed (excluding days with only one user).")
    
    # Print detailed statistics
    print_detailed_statistics(comparisons_cog, 'Cognitive')
    print_detailed_statistics(comparisons_social, 'Social')
    
    # Check if output directories exist, create if they don't
    if output_path:
        figures_dir = os.path.dirname(output_path)  # This should be outputs/figures
        tables_dir = os.path.join(os.path.dirname(figures_dir), 'tables')
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        
        # Generate daily statistics table
        tables_path = os.path.join(tables_dir, os.path.basename(output_path))
        daily_stats = create_daily_stats_table(agg_daily_cog, agg_daily_social, tables_path)
        print(f"Daily statistics table created and saved to {tables_path}_daily_stats.csv")
    else:
        daily_stats = create_daily_stats_table(agg_daily_cog, agg_daily_social)
    
    # Print the daily statistics table to the console
    print_daily_stats_table(daily_stats)
    
    # Create the combined visualization (this goes to figures directory)
    fig, axes = plot_combined_visualization(
        agg_daily_cog, 
        agg_daily_social,
        periods,
        daily_cognition=daily_cognition,
        daily_social=daily_social,
        period_stats_cog=period_stats_cog,
        period_stats_social=period_stats_social,
        comparisons_cog=comparisons_cog,
        comparisons_social=comparisons_social,
        output_path=f"{output_path}_combined_visualization.png" if output_path else None
    )
    
    print("Combined visualization generated.")
    
    if output_path:
        print(f"Combined figure saved to: {output_path}_combined_visualization.png")
    
    # Generate statistical results tables (these go to tables directory)
    tables_path = os.path.join(tables_dir, os.path.basename(output_path))
    period_stats_df, comparisons_df = create_statistical_results_table(
        period_stats_cog,
        period_stats_social,
        comparisons_cog,
        comparisons_social,
        output_path=tables_path if output_path else None
    )
    
    return df, (daily_cognition, daily_social), (
        (period_stats_cog, comparisons_cog), 
        (period_stats_social, comparisons_social)
    )

def print_daily_stats_table(daily_stats):
    """
    Print the daily statistics table to the console.
    
    Parameters:
    -----------
    daily_stats : pandas.DataFrame
        DataFrame containing the daily statistics
    """
    print("\n=== DAILY STATISTICS TABLE ===\n")
    print(f"{'Day':>5} | {'Cognitive Mean':>15} | {'Cognitive Users':>15} | {'Social Mean':>15} | {'Social Users':>15}")
    print(f"{'-'*5} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15}")
    
    for _, row in daily_stats.iterrows():
        day = row['Day']
        
        # Handle potentially NaN values
        if pd.isna(row['Cognitive Mean']):
            cog_mean_str = "N/A".rjust(15)
        else:
            cog_mean_str = f"{row['Cognitive Mean']:15.3f}"
            
        if pd.isna(row['Cognitive Users']):
            cog_users_str = "N/A".rjust(15)
        else:
            cog_users_str = f"{int(row['Cognitive Users']):15d}"
            
        if pd.isna(row['Social Mean']):
            social_mean_str = "N/A".rjust(15)
        else:
            social_mean_str = f"{row['Social Mean']:15.3f}"
            
        if pd.isna(row['Social Users']):
            social_users_str = "N/A".rjust(15)
        else:
            social_users_str = f"{int(row['Social Users']):15d}"
        
        print(f"{day:5.1f} | {cog_mean_str} | {cog_users_str} | {social_mean_str} | {social_users_str}")

if __name__ == "__main__":
    # Example usage
    sentence_level_path = "/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/final/sentence_level.csv"
    
    # Create output directory if it doesn't exist
    output_dir = "/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis with output path in the figures directory
    output_path = os.path.join(output_dir, "word_dynamics_results")
    
    df, daily_usage, analysis_results = analyze_word_dynamics(
        sentence_level_path,
        output_path=output_path
    )

    # Save the dataframes with daily word usage
    daily_cognition, daily_social = daily_usage
    daily_cognition.to_csv("/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/daily_cognition.csv", index=False)
    daily_social.to_csv("/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/daily_social.csv", index=False)
    
    print("\nAnalysis complete. Data files saved.")
    print(f"Figures saved to: {output_dir}")
