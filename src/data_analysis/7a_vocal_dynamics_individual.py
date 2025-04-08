# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests

def load_data(vocal_features_path):
    """
    Load the vocal features data for analysis.
    
    Parameters:
    -----------
    vocal_features_path : str
        Path to the vocal features CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Prepared dataframe
    """
    # Load vocal features data
    df = pd.read_csv(vocal_features_path)
    
    # Only include values where RelativeDate is between -14 and 14
    df = df[(df['RelativeDate'] >= -14) & (df['RelativeDate'] <= 14)].copy()
    
    # Make sure RecordingID is available (it might be called something else in your data)
    if 'RecordingID' not in df.columns:
        if 'Recording' in df.columns:
            df['RecordingID'] = df['Recording']
        elif 'recording_id' in df.columns:
            df['RecordingID'] = df['recording_id']
        else:
            # If no recording ID is available, use a combination of UserID and timestamp
            print("Warning: No RecordingID found. Creating a synthetic ID from user and date.")
            if 'Timestamp' in df.columns:
                df['RecordingID'] = df['UserID'].astype(str) + '_' + df['Timestamp'].astype(str)
            else:
                # Last resort: create a unique ID for each row
                print("Warning: No timestamp found. Each row will be treated as a unique recording.")
                df['RecordingID'] = np.arange(len(df))
    
    # Extract pitch mean from F0 semitone features
    # Convert from semitones (relative to 27.5Hz) to Hz
    df['PitchMeanHz'] = 27.5 * (2 ** (df['F0semitoneFrom27.5Hz_sma3nz_amean'] / 12))
    
    # Use standard deviation normalized as pitch variability
    df['PitchStdDevHz'] = df['F0semitoneFrom27.5Hz_sma3nz_stddevNorm']
    
    # Use jitter and shimmer features
    df['JitterLocal'] = df['jitterLocal_sma3nz_amean']
    df['ShimmerLocal'] = df['shimmerLocaldB_sma3nz_amean']
    
    return df

def filter_valid_users(df, min_entries=2):
    """
    Filter users who have at least a minimum number of entries in both pre and post periods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with vocal features
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

def calculate_daily_vocal_deviations(df, min_recordings=1):
    """
    Calculate standardized daily deviations from individual means for vocal features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with vocal features
    min_recordings : int, optional
        Minimum number of recordings required per day (default: 1)
        
    Returns:
    --------
    tuple
        Tuple of DataFrames with standardized daily deviations (pitch, jitter, shimmer)
    """
    # Calculate mean and standard deviation for each user's pitch
    user_pitch_stats = df.groupby('UserID')['PitchMeanHz'].agg(['mean', 'std']).reset_index()
    user_pitch_stats.columns = ['UserID', 'UserPitchMean', 'UserPitchStd']
    
    # Calculate mean and standard deviation for each user's jitter
    user_jitter_stats = df.groupby('UserID')['JitterLocal'].agg(['mean', 'std']).reset_index()
    user_jitter_stats.columns = ['UserID', 'UserJitterMean', 'UserJitterStd']
    
    # Calculate mean and standard deviation for each user's shimmer
    user_shimmer_stats = df.groupby('UserID')['ShimmerLocal'].agg(['mean', 'std']).reset_index()
    user_shimmer_stats.columns = ['UserID', 'UserShimmerMean', 'UserShimmerStd']
    
    # Merge user stats back to the original dataframe
    df_with_stats = df.merge(user_pitch_stats, on='UserID')
    df_with_stats = df_with_stats.merge(user_jitter_stats, on='UserID')
    df_with_stats = df_with_stats.merge(user_shimmer_stats, on='UserID')
    
    # Calculate z-scores (standardized deviations from individual mean)
    # Handle cases where standard deviation is 0 to avoid division by zero
    df_with_stats['PitchDeviation'] = np.where(
        df_with_stats['UserPitchStd'] > 0,
        (df_with_stats['PitchMeanHz'] - df_with_stats['UserPitchMean']) / df_with_stats['UserPitchStd'],
        0  # If std is 0, deviation is 0
    )
    
    df_with_stats['JitterDeviation'] = np.where(
        df_with_stats['UserJitterStd'] > 0,
        (df_with_stats['JitterLocal'] - df_with_stats['UserJitterMean']) / df_with_stats['UserJitterStd'],
        0  # If std is 0, deviation is 0
    )
    
    df_with_stats['ShimmerDeviation'] = np.where(
        df_with_stats['UserShimmerStd'] > 0,
        (df_with_stats['ShimmerLocal'] - df_with_stats['UserShimmerMean']) / df_with_stats['UserShimmerStd'],
        0  # If std is 0, deviation is 0
    )
    
    # Count unique recordings per user per day
    recording_counts = df_with_stats.groupby(['UserID', 'RelativeDate'])['RecordingID'].nunique().reset_index()
    recording_counts.columns = ['UserID', 'RelativeDate', 'RecordingCount']
    
    # Merge recording counts back to the dataframe
    df_with_stats = df_with_stats.merge(recording_counts, on=['UserID', 'RelativeDate'])
    
    # Filter based on the minimum recording threshold
    df_with_stats_filtered = df_with_stats[df_with_stats['RecordingCount'] >= min_recordings].copy()
    
    print(f"Filtered out {len(df_with_stats) - len(df_with_stats_filtered)} entries with fewer than {min_recordings} recording(s) per day.")
    print(f"Retained {len(df_with_stats_filtered)} entries with {min_recordings}+ recording(s) per day.")
    
    # Group by user and day to get average daily deviations (using only days with multiple recordings)
    daily_pitch = df_with_stats_filtered.groupby(['UserID', 'RelativeDate'])['PitchDeviation'].mean().reset_index()
    daily_jitter = df_with_stats_filtered.groupby(['UserID', 'RelativeDate'])['JitterDeviation'].mean().reset_index()
    daily_shimmer = df_with_stats_filtered.groupby(['UserID', 'RelativeDate'])['ShimmerDeviation'].mean().reset_index()
    
    return daily_pitch, daily_jitter, daily_shimmer

def aggregate_daily_vocal_deviations(daily_df, feature_col):
    """
    Aggregate daily vocal deviations across all users.
    
    Parameters:
    -----------
    daily_df : pandas.DataFrame
        Dataframe with daily vocal deviations per user
    feature_col : str
        Column name of the feature to analyze
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with aggregated daily deviations across all users
    """
    # Group by day to aggregate across all users
    agg_daily = daily_df.groupby('RelativeDate')[feature_col].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error
    agg_daily['se'] = agg_daily['std'] / np.sqrt(agg_daily['count'])
    
    # Create a dataframe with all days from -14 to 14
    all_days = pd.DataFrame({'RelativeDate': np.arange(-14, 15, 0.5 if 0.5 in daily_df['RelativeDate'].values else 1)})
    
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

def calculate_period_statistics(daily_df, feature_col):
    """
    Calculate statistics for different periods around the dosing.
    
    Parameters:
    -----------
    daily_df : pandas.DataFrame
        The dataframe with daily vocal deviations
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
        period_data = daily_df[
            (daily_df['RelativeDate'] >= start) & 
            (daily_df['RelativeDate'] < end)
        ]
        
        period_stats[period_name] = {
            'mean': period_data[feature_col].mean(),
            'std': period_data[feature_col].std(),
            'count': len(period_data),
            'se': period_data[feature_col].std() / np.sqrt(len(period_data)),
            'days': end - start
        }
    
    return period_stats, periods

def pairwise_period_comparisons(daily_df, periods, feature_col):
    """
    Perform pairwise t-tests between periods.
    
    Parameters:
    -----------
    daily_df : pandas.DataFrame
        Dataframe with daily vocal deviations
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
            period1_data = daily_df[
                (daily_df['RelativeDate'] >= start1) & 
                (daily_df['RelativeDate'] < end1)
            ][feature_col]
            
            # Get data for period 2
            start2, end2 = periods[period2]
            period2_data = daily_df[
                (daily_df['RelativeDate'] >= start2) & 
                (daily_df['RelativeDate'] < end2)
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
    print(f"\n--- {feature_name.upper()} DEVIATIONS COMPARISONS ---\n")
    
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

def plot_combined_time_series(agg_daily_pitch, agg_daily_jitter, agg_daily_shimmer, periods, 
                             min_day=-14, max_day=14, min_users=2,
                             daily_pitch=None, daily_jitter=None, daily_shimmer=None, output_path=None):
    """
    Plot time series of vocal feature deviations with consistent styling.
    """
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Filter data to only include days with at least min_users
    agg_daily_pitch_filtered = agg_daily_pitch[agg_daily_pitch['count'] >= min_users].copy()
    agg_daily_jitter_filtered = agg_daily_jitter[agg_daily_jitter['count'] >= min_users].copy()
    agg_daily_shimmer_filtered = agg_daily_shimmer[agg_daily_shimmer['count'] >= min_users].copy()
    
    # Create empty lists to store legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # Plot individual data points if provided (with reduced alpha)
    if daily_pitch is not None:
        pitch_points = ax.scatter(daily_pitch['RelativeDate'], daily_pitch['PitchDeviation'], 
                                color='#e41a1c', alpha=0.1, s=10)
        legend_handles.append(pitch_points)
        legend_labels.append('Individual Pitch Data')
    
    if daily_jitter is not None:
        jitter_points = ax.scatter(daily_jitter['RelativeDate'], daily_jitter['JitterDeviation'], 
                                 color='#377eb8', alpha=0.1, s=10)
        legend_handles.append(jitter_points)
        legend_labels.append('Individual Jitter Data')
    
    if daily_shimmer is not None:
        shimmer_points = ax.scatter(daily_shimmer['RelativeDate'], daily_shimmer['ShimmerDeviation'], 
                                  color='#4daf4a', alpha=0.1, s=10)
        legend_handles.append(shimmer_points)
        legend_labels.append('Individual Shimmer Data')
    
    # Plot mean lines with consistent colors
    if not agg_daily_pitch_filtered.empty:
        pitch_line = ax.plot(agg_daily_pitch_filtered['RelativeDate'], 
                           agg_daily_pitch_filtered['smoothed_mean'], 
                           color='#e41a1c', linewidth=2, label='Pitch')[0]
        legend_handles.append(pitch_line)
        legend_labels.append('Mean Pitch')
        
        # Add confidence interval
        ax.fill_between(
            agg_daily_pitch_filtered['RelativeDate'],
            agg_daily_pitch_filtered['smoothed_mean'] - agg_daily_pitch_filtered['se'],
            agg_daily_pitch_filtered['smoothed_mean'] + agg_daily_pitch_filtered['se'],
            color='#e41a1c', alpha=0.1
        )
    
    if not agg_daily_jitter_filtered.empty:
        jitter_line = ax.plot(agg_daily_jitter_filtered['RelativeDate'], 
                            agg_daily_jitter_filtered['smoothed_mean'], 
                            color='#377eb8', linewidth=2, label='Jitter')[0]
        legend_handles.append(jitter_line)
        legend_labels.append('Mean Jitter')
        
        # Add confidence interval
        ax.fill_between(
            agg_daily_jitter_filtered['RelativeDate'],
            agg_daily_jitter_filtered['smoothed_mean'] - agg_daily_jitter_filtered['se'],
            agg_daily_jitter_filtered['smoothed_mean'] + agg_daily_jitter_filtered['se'],
            color='#377eb8', alpha=0.1
        )
    
    if not agg_daily_shimmer_filtered.empty:
        shimmer_line = ax.plot(agg_daily_shimmer_filtered['RelativeDate'], 
                             agg_daily_shimmer_filtered['smoothed_mean'], 
                             color='#4daf4a', linewidth=2, label='Shimmer')[0]
        legend_handles.append(shimmer_line)
        legend_labels.append('Mean Shimmer')
        
        # Add confidence interval
        ax.fill_between(
            agg_daily_shimmer_filtered['RelativeDate'],
            agg_daily_shimmer_filtered['smoothed_mean'] - agg_daily_shimmer_filtered['se'],
            agg_daily_shimmer_filtered['smoothed_mean'] + agg_daily_shimmer_filtered['se'],
            color='#4daf4a', alpha=0.1
        )
    
    # Add vertical line at day 0 (dosing)
    dosing_line = ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    legend_handles.append(dosing_line)
    legend_labels.append('Dosing Day')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add labels and title with consistent font sizes
    ax.set_xlabel('Days Relative to Dosing', fontsize=12)
    ax.set_ylabel('Standardized Deviation from Individual Mean', fontsize=12)
    ax.set_title('Temporal Evolution of Vocal Features Around 5-MeO-DMT Dosing', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.2)
    
    # Set x-axis limits
    ax.set_xlim(min_day, max_day)
    
    # Add legend inside the plot
    ax.legend(legend_handles, legend_labels, 
             loc='upper right',  # Position in upper right corner
             frameon=True,       # Add frame
             framealpha=0.9,     # Make frame slightly transparent
             edgecolor='none',   # No edge color for frame
             fontsize=10)        # Slightly smaller font size
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_period_comparison(period_stats_pitch, period_stats_jitter, period_stats_shimmer, 
                          comparisons_pitch, comparisons_jitter, comparisons_shimmer, 
                          output_path=None):
    """
    Plot bar chart comparing vocal feature deviations across periods with consistent styling.
    """
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Function to plot one feature's comparison with consistent styling
    def plot_feature_comparison(ax, period_stats, comparisons, title, color):
        periods = list(period_stats.keys())
        means = [stats['mean'] for stats in period_stats.values()]
        errors = [stats['se'] for stats in period_stats.values()]
        
        # Create bars with consistent styling
        bars = ax.bar(range(len(periods)), means, yerr=errors, capsize=5, 
                     color=color, alpha=0.7, edgecolor='none')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add significance bars
        y_pos = max(means) + max(errors) * 1.2
        bar_height = (max(means) - min(means)) * 0.05
        
        for comp in comparisons:
            if comp['p-value-fdr'] < 0.05:
                idx1 = periods.index(comp['Period 1'])
                idx2 = periods.index(comp['Period 2'])
                
                # Draw the line
                ax.plot([idx1, idx2], [y_pos, y_pos], color='black', linewidth=1)
                
                # Add significance stars
                if comp['p-value-fdr'] < 0.001:
                    stars = '***'
                elif comp['p-value-fdr'] < 0.01:
                    stars = '**'
                else:
                    stars = '*'
                    
                # Increased font size for significance stars
                ax.text((idx1 + idx2)/2, y_pos + bar_height*0.1, stars, 
                        ha='center', va='bottom', fontsize=12)  # Increased from 10
                
                y_pos += bar_height * 1.5
        
        # Set y-axis limit to accommodate significance bars
        ax.set_ylim(min(min(means) - max(errors) * 1.2, 0), 
                    y_pos + bar_height)
        
        # Add labels and title with increased font sizes
        ax.set_xlabel('Time Period', fontsize=14)  # Increased from 12
        ax.set_ylabel('Mean Standardized Deviation', fontsize=14)  # Increased from 12
        ax.set_title(title, fontsize=16)  # Increased from 14
        
        # Set x-tick labels with increased font size
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels([p.split(' (')[0] for p in periods], 
                          rotation=45, ha='right', fontsize=12)  # Increased from 10
        
        # Increase y-axis tick label font size
        ax.tick_params(axis='y', labelsize=12)  # Increased from 10
        
        # Add grid
        ax.grid(True, alpha=0.2, axis='y')
    
    # Plot comparisons for each feature with consistent colors
    plot_feature_comparison(ax1, period_stats_pitch, comparisons_pitch, 
                          'Pitch', '#e41a1c')
    plot_feature_comparison(ax2, period_stats_jitter, comparisons_jitter, 
                          'Jitter', '#377eb8')
    plot_feature_comparison(ax3, period_stats_shimmer, comparisons_shimmer, 
                          'Shimmer', '#4daf4a')
    
    # Adjust layout to prevent text cutoff with the larger font sizes
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
    
    return fig, (ax1, ax2, ax3)

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

def analyze_standardized_vocal_dynamics(vocal_features_path, min_recordings=1, min_users=2, output_path=None):
    """
    Perform complete standardized vocal dynamics analysis.
    
    Parameters:
    -----------
    vocal_features_path : str
        Path to the vocal features CSV file
    min_recordings : int, optional
        Minimum number of recordings required per day (default: 1)
    min_users : int, optional
        Minimum number of users required for a day to be plotted (default: 2)
    output_path : str, optional
        Path to save the output figure
        
    Returns:
    --------
    tuple
        Tuple containing the dataframe with standardized deviations and analysis results
    """
    print("\n=== RUNNING STANDARDIZED VOCAL DYNAMICS ANALYSIS ===\n")
    
    # Load data
    df = load_data(vocal_features_path)
    print(f"Data loaded successfully. {len(df)} entries found.")
    
    # Filter valid users
    df = filter_valid_users(df)
    print(f"Filtered valid users: {df['UserID'].nunique()} users remaining.")
    
    # Add period column
    df = add_period_column(df)
    print("Period column added.")
    
    # Calculate daily vocal deviations with minimum recording threshold
    daily_pitch, daily_jitter, daily_shimmer = calculate_daily_vocal_deviations(df, min_recordings=min_recordings)
    print(f"Daily vocal deviations calculated (min. {min_recordings} recording(s) per day).")
    
    # Aggregate daily vocal deviations
    agg_daily_pitch = aggregate_daily_vocal_deviations(daily_pitch, 'PitchDeviation')
    agg_daily_jitter = aggregate_daily_vocal_deviations(daily_jitter, 'JitterDeviation')
    agg_daily_shimmer = aggregate_daily_vocal_deviations(daily_shimmer, 'ShimmerDeviation')
    print("Daily vocal deviations aggregated across users.")
    
    # Calculate period statistics for pitch
    period_stats_pitch, periods = calculate_period_statistics(daily_pitch, 'PitchDeviation')
    
    # Calculate period statistics for jitter
    period_stats_jitter, _ = calculate_period_statistics(daily_jitter, 'JitterDeviation')
    
    # Calculate period statistics for shimmer
    period_stats_shimmer, _ = calculate_period_statistics(daily_shimmer, 'ShimmerDeviation')
    print("Period statistics calculated.")
    
    # Perform pairwise comparisons for pitch
    comparisons_pitch = pairwise_period_comparisons(daily_pitch, periods, 'PitchDeviation')
    
    # Perform pairwise comparisons for jitter
    comparisons_jitter = pairwise_period_comparisons(daily_jitter, periods, 'JitterDeviation')
    
    # Perform pairwise comparisons for shimmer
    comparisons_shimmer = pairwise_period_comparisons(daily_shimmer, periods, 'ShimmerDeviation')
    print("Pairwise comparisons completed.")
    
    # Print detailed statistics
    print_detailed_statistics(comparisons_pitch, 'Pitch')
    print_detailed_statistics(comparisons_jitter, 'Jitter')
    print_detailed_statistics(comparisons_shimmer, 'Shimmer')
    
    # Create a table with daily values for all features
    feature_table, min_day, max_day = create_vocal_features_table(
        agg_daily_pitch,
        agg_daily_jitter,
        agg_daily_shimmer,
        min_users=min_users,
        output_path=output_path
    )
    print(f"\nTable of daily vocal feature deviations created for days {min_day} to {max_day}.")
    print(f"Only days with at least {min_users} users are included in plots.")
    
    # Update output paths to use outputs/figures directory
    base_output_path = f"{output_path}/outputs/figures/vocal_dynamics" if output_path else None
    
    # Plot combined time series with individual data points
    time_fig, time_ax = plot_combined_time_series(
        agg_daily_pitch, 
        agg_daily_jitter,
        agg_daily_shimmer,
        periods,
        min_day=min_day,
        max_day=max_day,
        min_users=min_users,
        daily_pitch=daily_pitch,
        daily_jitter=daily_jitter,
        daily_shimmer=daily_shimmer,
        output_path=f"{base_output_path}_timeseries.png" if base_output_path else None
    )
    
    # Create a version without individual points (cleaner visualization)
    time_fig_clean, time_ax_clean = plot_combined_time_series(
        agg_daily_pitch, 
        agg_daily_jitter,
        agg_daily_shimmer,
        periods,
        min_day=min_day,
        max_day=max_day,
        min_users=min_users,
        output_path=f"{base_output_path}_timeseries_clean.png" if base_output_path else None
    )
    
    # Plot period comparisons
    bar_fig, bar_axes = plot_period_comparison(
        period_stats_pitch,
        period_stats_jitter,
        period_stats_shimmer,
        comparisons_pitch,
        comparisons_jitter,
        comparisons_shimmer,
        output_path=f"{base_output_path}_barplot.png" if base_output_path else None
    )
    
    print("Plots generated.")
    
    if output_path:
        print(f"Figures saved with base name: {base_output_path}")
    
    # Create result dataframes
    results = {
        'daily_pitch': daily_pitch,
        'daily_jitter': daily_jitter,
        'daily_shimmer': daily_shimmer,
        'agg_daily_pitch': agg_daily_pitch,
        'agg_daily_jitter': agg_daily_jitter,
        'agg_daily_shimmer': agg_daily_shimmer,
        'period_stats_pitch': period_stats_pitch,
        'period_stats_jitter': period_stats_jitter,
        'period_stats_shimmer': period_stats_shimmer,
        'comparisons_pitch': comparisons_pitch,
        'comparisons_jitter': comparisons_jitter,
        'comparisons_shimmer': comparisons_shimmer,
        'feature_table': feature_table
    }
    
    return df, results

def create_vocal_features_table(agg_daily_pitch, agg_daily_jitter, agg_daily_shimmer, min_users=1, output_path=None):
    """
    Create a table summarizing the daily mean values for all three vocal features.
    
    Parameters:
    -----------
    agg_daily_pitch, agg_daily_jitter, agg_daily_shimmer : pandas.DataFrame
        Dataframes with aggregated daily vocal deviations
    min_users : int, optional
        Minimum number of users required for analysis (default: 1)
    output_path : str, optional
        Path to save the output table
        
    Returns:
    --------
    tuple
        (table_df, min_day, max_day) - The table dataframe and the day range
    """
    # Find the min and max days with sufficient data
    min_day, max_day = -14, 14  # Default range
    
    # Determine the actual range of days with sufficient data
    for df in [agg_daily_pitch, agg_daily_jitter, agg_daily_shimmer]:
        valid_days = df[(~df['smoothed_mean'].isna()) & (df['count'] >= min_users)]['RelativeDate']
        if not valid_days.empty:
            min_day = max(min_day, int(valid_days.min()))
            max_day = min(max_day, int(valid_days.max()))
    
    # Create a dataframe with the adjusted day range
    days = np.arange(min_day, max_day + 1)
    table_df = pd.DataFrame({'Day': days})
    
    # Extract the mean values and counts for each feature and merge into the table
    pitch_data = agg_daily_pitch[['RelativeDate', 'smoothed_mean', 'count']].rename(
        columns={'RelativeDate': 'Day', 'smoothed_mean': 'Pitch Deviation', 'count': 'Pitch Count'})
    
    jitter_data = agg_daily_jitter[['RelativeDate', 'smoothed_mean', 'count']].rename(
        columns={'RelativeDate': 'Day', 'smoothed_mean': 'Jitter Deviation', 'count': 'Jitter Count'})
    
    shimmer_data = agg_daily_shimmer[['RelativeDate', 'smoothed_mean', 'count']].rename(
        columns={'RelativeDate': 'Day', 'smoothed_mean': 'Shimmer Deviation', 'count': 'Shimmer Count'})
    
    # Merge all features into one table
    table_df = pd.merge(table_df, pitch_data, on='Day', how='left')
    table_df = pd.merge(table_df, jitter_data, on='Day', how='left')
    table_df = pd.merge(table_df, shimmer_data, on='Day', how='left')
    
    # Round deviations to 3 decimal places for better readability
    for col in ['Pitch Deviation', 'Jitter Deviation', 'Shimmer Deviation']:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(3)
    
    # Instead of summing the counts (which gives 3x the actual number)
    # Just use one of the counts since they're all the same
    table_df['User Count'] = table_df['Pitch Count']
    
    # Or verify they're the same and then use one of them
    # This is a more robust approach
    mask = ~table_df['Pitch Count'].isna()
    if mask.any():
        # Verify counts are the same across features
        count_matches = (table_df.loc[mask, 'Pitch Count'] == 
                        table_df.loc[mask, 'Jitter Count']).all() and \
                        (table_df.loc[mask, 'Pitch Count'] == 
                        table_df.loc[mask, 'Shimmer Count']).all()
        
        if count_matches:
            table_df['User Count'] = table_df['Pitch Count']
            # Convert to int
            table_df.loc[mask, 'User Count'] = table_df.loc[mask, 'User Count'].astype(int)
        else:
            # If they differ for some reason, take the average
            table_df['User Count'] = table_df[['Pitch Count', 'Jitter Count', 
                                            'Shimmer Count']].mean(axis=1).round().astype('Int64')
            print("Warning: Feature counts differ for some days. Using average count.")
    
    # Drop the Total Count column if it exists
    if 'Total Count' in table_df.columns:
        table_df = table_df.drop(columns=['Total Count'])
    
    # Highlight the dosing day (day 0)
    # This only works for display in notebooks or when saved as HTML
    try:
        # Find the index for day 0
        day_zero_idx = table_df.index[table_df['Day'] == 0].tolist()
        if day_zero_idx:
            table_df = table_df.style.apply(
                lambda x: ['background-color: #ffffcc' if i == day_zero_idx[0] else '' for i in range(len(x))], 
                axis=0
            )
    except:
        # If styling fails, continue with the unstylized dataframe
        pass
    
    # Save to CSV if output path is provided
    if output_path:
        # If we have a styled dataframe, we need to get the underlying data
        if hasattr(table_df, 'data'):
            table_df.data.to_csv(f"{output_path}_table.csv", index=False)
        else:
            table_df.to_csv(f"{output_path}_table.csv", index=False)
    
    # Add a note to indicate days with insufficient data
    if min_users > 1:
        # Mark days with fewer users than min_users
        cols_to_check = ['User Count']
        if 'User Count' in table_df.columns:
            # If we have a styled dataframe
            if hasattr(table_df, 'data'):
                insufficient_mask = table_df.data['User Count'] < min_users
                insufficient_days = table_df.data.loc[insufficient_mask, 'Day'].tolist()
            else:
                insufficient_mask = table_df['User Count'] < min_users
                insufficient_days = table_df.loc[insufficient_mask, 'Day'].tolist()
            
            if insufficient_days:
                print(f"Note: Days {insufficient_days} have fewer than {min_users} users and are not included in the plot.")
    
    # Return the table and the day range
    return table_df, min_day, max_day

if __name__ == "__main__":
    # Example usage
    CORE_DIR = "/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals"
    vocal_features_path = f"{CORE_DIR}/data/processed/vocal_features.csv"
    
    # Run analysis with updated output path
    df, results = analyze_standardized_vocal_dynamics(
        vocal_features_path,
        output_path=CORE_DIR  # Pass the core directory
    )

    # Save the dataframes with daily vocal deviations
    results['daily_pitch'].to_csv(f"{CORE_DIR}/data/processed/daily_pitch_deviations.csv", index=False)
    results['daily_jitter'].to_csv(f"{CORE_DIR}/data/processed/daily_jitter_deviations.csv", index=False)
    results['daily_shimmer'].to_csv(f"{CORE_DIR}/data/processed/daily_shimmer_deviations.csv", index=False)
    
    # Print the table to console (limited to first few rows for preview)
    if 'feature_table' in results:
        print("\nPreview of daily vocal feature deviations:")
        try:
            print(results['feature_table'].head(10))
        except:
            # If it's a styled dataframe, access the underlying data
            if hasattr(results['feature_table'], 'data'):
                print(results['feature_table'].data.head(10))

