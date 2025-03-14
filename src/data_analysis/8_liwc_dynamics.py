# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests

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

def calculate_period_statistics(daily_usage_df, feature_col):
    """
    Calculate statistics for different periods around the dosing.
    
    Parameters:
    -----------
    daily_usage_df : pandas.DataFrame
        The dataframe with daily word usage values
    feature_col : str
        Column name of the feature to analyze
        
    Returns:
    --------
    dict
        Dictionary with period statistics
    tuple
        Tuple containing the periods dictionary
    """
    # Define periods as weekly ranges
    periods = {
        'Week -2 (-14 to -7)': (-14, -7),
        'Week -1 (-7 to 0)': (-7, 0),
        'Week +1 (0 to 7)': (0, 7),
        'Week +2 (7 to 14)': (7, 14)
    }
    
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

def pairwise_period_comparisons(daily_usage_df, periods, feature_col):
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
        
    Returns:
    --------
    list
        List of dictionaries with comparison results
    """
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

def plot_combined_time_series(agg_daily_cog, agg_daily_social, periods, output_path=None):
    """
    Plot time series of cognitive and social word usage with period shading and a break at dosing time.
    
    Parameters:
    -----------
    agg_daily_cog : pandas.DataFrame
        Dataframe with aggregated daily cognitive word usage
    agg_daily_social : pandas.DataFrame
        Dataframe with aggregated daily social word usage
    periods : dict
        Dictionary with period definitions
    output_path : str, optional
        Path to save the output figure
    
    Returns:
    --------
    tuple
        Tuple containing the figure and axis
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Split data into pre and post dosing segments to create a break for cognitive words
    # Filter out NaN values to avoid plotting gaps
    pre_dosing_cog = agg_daily_cog[(agg_daily_cog['RelativeDate'] <= -0.5) & (~agg_daily_cog['smoothed_mean'].isna())]
    post_dosing_cog = agg_daily_cog[(agg_daily_cog['RelativeDate'] >= 0.5) & (~agg_daily_cog['smoothed_mean'].isna())]
    
    # Split data into pre and post dosing segments to create a break for social words
    # Filter out NaN values to avoid plotting gaps
    pre_dosing_social = agg_daily_social[(agg_daily_social['RelativeDate'] <= -0.5) & (~agg_daily_social['smoothed_mean'].isna())]
    post_dosing_social = agg_daily_social[(agg_daily_social['RelativeDate'] >= 0.5) & (~agg_daily_social['smoothed_mean'].isna())]
    
    # Plot pre-dosing segment for cognitive words
    if not pre_dosing_cog.empty:
        ax.plot(pre_dosing_cog['RelativeDate'], pre_dosing_cog['smoothed_mean'], 
                color='purple', linewidth=2, label='Cognitive Word Usage')
        
        # Add confidence interval for pre-dosing cognitive words
        ax.fill_between(
            pre_dosing_cog['RelativeDate'],
            pre_dosing_cog['smoothed_mean'] - pre_dosing_cog['se'],
            pre_dosing_cog['smoothed_mean'] + pre_dosing_cog['se'],
            color='purple', alpha=0.2
        )
    
    # Plot post-dosing segment for cognitive words
    if not post_dosing_cog.empty:
        ax.plot(post_dosing_cog['RelativeDate'], post_dosing_cog['smoothed_mean'], 
                color='purple', linewidth=2, label=None if not pre_dosing_cog.empty else 'Cognitive Word Usage')
        
        # Add confidence interval for post-dosing cognitive words
        ax.fill_between(
            post_dosing_cog['RelativeDate'],
            post_dosing_cog['smoothed_mean'] - post_dosing_cog['se'],
            post_dosing_cog['smoothed_mean'] + post_dosing_cog['se'],
            color='purple', alpha=0.2
        )
    
    # Plot pre-dosing segment for social words
    if not pre_dosing_social.empty:
        ax.plot(pre_dosing_social['RelativeDate'], pre_dosing_social['smoothed_mean'], 
                color='green', linewidth=2, label='Social Word Usage')
        
        # Add confidence interval for pre-dosing social words
        ax.fill_between(
            pre_dosing_social['RelativeDate'],
            pre_dosing_social['smoothed_mean'] - pre_dosing_social['se'],
            pre_dosing_social['smoothed_mean'] + pre_dosing_social['se'],
            color='green', alpha=0.2
        )
    
    # Plot post-dosing segment for social words
    if not post_dosing_social.empty:
        ax.plot(post_dosing_social['RelativeDate'], post_dosing_social['smoothed_mean'], 
                color='green', linewidth=2, label=None if not pre_dosing_social.empty else 'Social Word Usage')
        
        # Add confidence interval for post-dosing social words
        ax.fill_between(
            post_dosing_social['RelativeDate'],
            post_dosing_social['smoothed_mean'] - post_dosing_social['se'],
            post_dosing_social['smoothed_mean'] + post_dosing_social['se'],
            color='green', alpha=0.2
        )
    
    # Add vertical line at day 0 (dosing)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Dosing Day')
    
    # Define colors for periods
    period_colors = {
        'Week -2 (-14 to -7)': 'lightgray',
        'Week -1 (-7 to 0)': 'lightgreen',
        'Week +1 (0 to 7)': 'paleturquoise',
        'Week +2 (7 to 14)': 'lavender'
    }
    
    # Add period shading
    for period_name, (start, end) in periods.items():
        ax.axvspan(start, end, alpha=0.3, color=period_colors[period_name], label=period_name)
    
    # Add labels and title
    ax.set_xlabel('Days Relative to Dosing', fontsize=12)
    ax.set_ylabel('Word Usage (Standard Deviations from Individual Mean)', fontsize=12)
    ax.set_title('Temporal Evolution of Cognitive and Social Word Usage Around 5-MeO-DMT Dosing', fontsize=14)
    
    # Add horizontal line at y=0 to represent individual baseline
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Find and mark peak days, but only if we have valid data
    if not agg_daily_cog['smoothed_mean'].isna().all():
        valid_cog = agg_daily_cog[~agg_daily_cog['smoothed_mean'].isna()]
        cog_peak_date = valid_cog.loc[valid_cog['smoothed_mean'].idxmax(), 'RelativeDate']
        cog_peak_value = valid_cog['smoothed_mean'].max()
        ax.scatter(cog_peak_date, cog_peak_value, color='purple', s=100, zorder=5, 
                  label=f'Cognitive Peak (Day {cog_peak_date})')
    
    if not agg_daily_social['smoothed_mean'].isna().all():
        valid_social = agg_daily_social[~agg_daily_social['smoothed_mean'].isna()]
        social_peak_date = valid_social.loc[valid_social['smoothed_mean'].idxmax(), 'RelativeDate']
        social_peak_value = valid_social['smoothed_mean'].max()
        ax.scatter(social_peak_date, social_peak_value, color='green', s=100, zorder=5, 
                  label=f'Social Peak (Day {social_peak_date})')
    
    # Add legend - place it outside the plot for better visibility
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Set x-axis limits to focus on the period of interest
    ax.set_xlim(-14.5, 14.5)
    
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
    
    return fig, ax

def plot_period_comparison(period_stats_cog, period_stats_social, comparisons_cog, comparisons_social, output_path=None):
    """
    Plot bar chart comparing cognitive and social word usage across periods.
    
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
        Path to save the output figure
    
    Returns:
    --------
    tuple
        Tuple containing the figure and axes
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot for cognitive words
    periods = list(period_stats_cog.keys())
    means_cog = [stats['mean'] for stats in period_stats_cog.values()]
    errors_cog = [stats['se'] for stats in period_stats_cog.values()]
    
    # Create color array based on values (positive or negative)
    colors_cog = ['purple']
    
    bars1 = ax1.bar(range(len(periods)), means_cog, yerr=errors_cog, capsize=5, 
                   color=colors_cog, alpha=0.7, edgecolor='purple')
    
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
            idx1 = periods.index(comp['Period 1'])
            idx2 = periods.index(comp['Period 2'])
            
            # Draw the line
            ax1.plot([idx1, idx2], [y_pos, y_pos], color='black', linewidth=1.5)
            
            # Add significance stars
            if comp['p-value-fdr'] < 0.001:
                stars = '***'
            elif comp['p-value-fdr'] < 0.01:
                stars = '**'
            else:
                stars = '*'
                
            ax1.text((idx1 + idx2)/2, y_pos + bar_height*0.2, stars, 
                    ha='center', va='bottom', color='black')
            
            y_pos += bar_height * 1.5  # Increment for next bar
    
    # Add horizontal line at y=0
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Set y-axis limit to accommodate significance bars and include zero
    buffer = (y_pos - max_y) * 0.2  # Add a small buffer
    ax1.set_ylim(min_y - buffer, y_pos + buffer)
    
    # Add labels and title for cognitive words
    ax1.set_xlabel('Time Period Relative to Dosing', fontsize=12)
    ax1.set_ylabel('Mean Cognitive Word Usage\n(Standard Deviations from Individual Mean)', fontsize=12)
    ax1.set_title('Comparison of Cognitive Word Usage', fontsize=14)
    
    # Set x-tick labels
    ax1.set_xticks(range(len(periods)))
    ax1.set_xticklabels(periods, rotation=45, ha='right')
    
    # Add grid
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot for social words
    means_social = [stats['mean'] for stats in period_stats_social.values()]
    errors_social = [stats['se'] for stats in period_stats_social.values()]
    
    # Create color array based on values (positive or negative)
    colors_social = ['darkgreen']
    
    bars2 = ax2.bar(range(len(periods)), means_social, yerr=errors_social, capsize=5, 
                   color=colors_social, alpha=0.7, edgecolor='green')
    
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
            idx1 = periods.index(comp['Period 1'])
            idx2 = periods.index(comp['Period 2'])
            
            # Draw the line
            ax2.plot([idx1, idx2], [y_pos, y_pos], color='black', linewidth=1.5)
            
            # Add significance stars
            if comp['p-value-fdr'] < 0.001:
                stars = '***'
            elif comp['p-value-fdr'] < 0.01:
                stars = '**'
            else:
                stars = '*'
                
            ax2.text((idx1 + idx2)/2, y_pos + bar_height*0.2, stars, 
                    ha='center', va='bottom', color='black')
            
            y_pos += bar_height * 1.5  # Increment for next bar
    
    # Add horizontal line at y=0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Set y-axis limit to accommodate significance bars and include zero
    buffer = (y_pos - max_y) * 0.2  # Add a small buffer
    ax2.set_ylim(min_y - buffer, y_pos + buffer)
    
    # Add labels and title for social words
    ax2.set_xlabel('Time Period Relative to Dosing', fontsize=12)
    ax2.set_ylabel('Mean Social Word Usage\n(Standard Deviations from Individual Mean)', fontsize=12)
    ax2.set_title('Comparison of Social Word Usage', fontsize=14)
    
    # Set x-tick labels
    ax2.set_xticks(range(len(periods)))
    ax2.set_xticklabels(periods, rotation=45, ha='right')
    
    # Add grid
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
    
    return fig, (ax1, ax2)

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
    
    # Calculate period statistics for cognitive words
    period_stats_cog, periods = calculate_period_statistics(daily_cognition, 'CogDeviation')
    
    # Calculate period statistics for social words
    period_stats_social, _ = calculate_period_statistics(daily_social, 'SocialDeviation')
    print("Period statistics calculated.")
    
    # Perform pairwise comparisons for cognitive words
    comparisons_cog = pairwise_period_comparisons(daily_cognition, periods, 'CogDeviation')
    
    # Perform pairwise comparisons for social words
    comparisons_social = pairwise_period_comparisons(daily_social, periods, 'SocialDeviation')
    print("Pairwise comparisons completed.")
    
    # Print detailed statistics
    print_detailed_statistics(comparisons_cog, 'Cognitive')
    print_detailed_statistics(comparisons_social, 'Social')
    
    # Plot combined time series
    time_fig, time_ax = plot_combined_time_series(
        agg_daily_cog, 
        agg_daily_social,
        periods, 
        output_path=f"{output_path}_timeseries.png" if output_path else None
    )
    
    # Plot period comparisons
    bar_fig, bar_axes = plot_period_comparison(
        period_stats_cog,
        period_stats_social,
        comparisons_cog,
        comparisons_social,
        output_path=f"{output_path}_barplot.png" if output_path else None
    )
    
    print("Plots generated.")
    
    if output_path:
        print(f"Figures saved with base name: {output_path}")
    
    return df, (daily_cognition, daily_social), (
        (period_stats_cog, comparisons_cog), 
        (period_stats_social, comparisons_social)
    )

if __name__ == "__main__":
    # Example usage
    sentence_level_path = "/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/final/sentence_level.csv"
    
    # Run analysis
    df, daily_usage, analysis_results = analyze_word_dynamics(
        sentence_level_path,
        output_path="word_dynamics_results"
    )

    # Save the dataframes with daily word usage
    daily_cognition, daily_social = daily_usage
    daily_cognition.to_csv("/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/daily_cognition.csv", index=False)
    daily_social.to_csv("/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/daily_social.csv", index=False)
