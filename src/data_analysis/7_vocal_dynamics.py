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
    
    # Extract pitch mean from F0 semitone features
    # Convert from semitones (relative to 27.5Hz) to Hz
    df['PitchMeanHz'] = 27.5 * (2 ** (df['F0semitoneFrom27.5Hz_sma3nz_amean'] / 12))
    
    # Use standard deviation normalized as pitch variability
    df['PitchStdDevHz'] = df['F0semitoneFrom27.5Hz_sma3nz_stddevNorm']
    
    # Use jitter and shimmer features
    df['JitterLocal'] = df['jitterLocal_sma3nz_amean']
    df['ShimmerLocal'] = df['shimmerLocaldB_sma3nz_amean']
    
    return df

def filter_valid_users(df):
    """
    Filter users who have at least 2 recordings in both pre and post periods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with vocal features
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe with only valid users
    """
    # Find users who have at least 2 recordings in PrePost = 0
    pre_users = df[df['PrePost'] == 0]['UserID'].value_counts()
    pre_users = pre_users[pre_users >= 2].index.tolist()
    
    # Find users who have at least 2 recordings in PrePost = 1
    post_users = df[df['PrePost'] == 1]['UserID'].value_counts()
    post_users = post_users[post_users >= 2].index.tolist()
    
    # Find users who have at least 2 recordings in both periods
    valid_users = set(pre_users) & set(post_users)
    
    # Filter the dataframe to only include valid users
    return df[df['UserID'].isin(valid_users)]

def identify_vocal_deviations(df):
    """
    Identify deviations from the mean for pitch, jitter, and shimmer for each user.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with vocal features
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added columns for vocal deviations
    """
    # Calculate mean and deviations for pitch
    df['MeanPitch'] = df.groupby('UserID')['PitchMeanHz'].transform('mean')
    df['PitchDeviation'] = df['PitchMeanHz'] - df['MeanPitch']
    df['PitchDeviationStd'] = df.groupby('UserID')['PitchDeviation'].transform('std')
    
    # Classify significant pitch deviations
    df['PitchSignificantDeviation'] = (
        (df['PitchDeviation'] > df['PitchDeviationStd']) | 
        (df['PitchDeviation'] < -df['PitchDeviationStd'])
    )
    df['PitchDeviationType'] = 'None'
    df.loc[df['PitchSignificantDeviation'] & (df['PitchDeviation'] > df['PitchDeviationStd']), 'PitchDeviationType'] = 'High'
    df.loc[df['PitchSignificantDeviation'] & (df['PitchDeviation'] < -df['PitchDeviationStd']), 'PitchDeviationType'] = 'Low'
    
    # Calculate mean and deviations for jitter
    df['MeanJitter'] = df.groupby('UserID')['JitterLocal'].transform('mean')
    df['JitterDeviation'] = df['JitterLocal'] - df['MeanJitter']
    df['JitterDeviationStd'] = df.groupby('UserID')['JitterDeviation'].transform('std')
    
    # Classify significant jitter deviations
    df['JitterSignificantDeviation'] = (
        (df['JitterDeviation'] > df['JitterDeviationStd']) | 
        (df['JitterDeviation'] < -df['JitterDeviationStd'])
    )
    df['JitterDeviationType'] = 'None'
    df.loc[df['JitterSignificantDeviation'] & (df['JitterDeviation'] > df['JitterDeviationStd']), 'JitterDeviationType'] = 'High'
    df.loc[df['JitterSignificantDeviation'] & (df['JitterDeviation'] < -df['JitterDeviationStd']), 'JitterDeviationType'] = 'Low'
    
    # Calculate mean and deviations for shimmer
    df['MeanShimmer'] = df.groupby('UserID')['ShimmerLocal'].transform('mean')
    df['ShimmerDeviation'] = df['ShimmerLocal'] - df['MeanShimmer']
    df['ShimmerDeviationStd'] = df.groupby('UserID')['ShimmerDeviation'].transform('std')
    
    # Classify significant shimmer deviations
    df['ShimmerSignificantDeviation'] = (
        (df['ShimmerDeviation'] > df['ShimmerDeviationStd']) | 
        (df['ShimmerDeviation'] < -df['ShimmerDeviationStd'])
    )
    df['ShimmerDeviationType'] = 'None'
    df.loc[df['ShimmerSignificantDeviation'] & (df['ShimmerDeviation'] > df['ShimmerDeviationStd']), 'ShimmerDeviationType'] = 'High'
    df.loc[df['ShimmerSignificantDeviation'] & (df['ShimmerDeviation'] < -df['ShimmerDeviationStd']), 'ShimmerDeviationType'] = 'Low'
    
    return df

def calculate_daily_deviation_counts(df):
    """
    Calculate daily deviation counts for pitch, jitter, and shimmer across all users.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with identified vocal deviations
        
    Returns:
    --------
    tuple
        Tuple containing dataframes with daily deviation counts for pitch, jitter, and shimmer
    """
    # Ensure RelativeDate is integer
    df['RelativeDate'] = df['RelativeDate'].astype(int)
    
    # Function to calculate counts for a specific feature
    def calculate_counts(feature):
        # Count high deviations per day
        high_daily = (df[df[f'{feature}DeviationType'] == 'High']
                     .groupby('RelativeDate')
                     .size()
                     .reset_index(name='High'))
        
        # Count low deviations per day
        low_daily = (df[df[f'{feature}DeviationType'] == 'Low']
                    .groupby('RelativeDate')
                    .size()
                    .reset_index(name='Low'))
        
        # Merge the counts
        daily_counts = pd.merge(high_daily, low_daily, 
                               on='RelativeDate', 
                               how='outer').fillna(0)
        
        # Ensure we have all days from -14 to 14
        all_days = pd.DataFrame({'RelativeDate': range(-14, 15)})
        daily_counts = pd.merge(all_days, daily_counts, 
                               on='RelativeDate', 
                               how='left').fillna(0)
        
        # Sort by RelativeDate
        daily_counts = daily_counts.sort_values('RelativeDate')
        
        # Apply Gaussian smoothing
        daily_counts['SmoothedHigh'] = gaussian_filter1d(daily_counts['High'], sigma=1)
        daily_counts['SmoothedLow'] = gaussian_filter1d(daily_counts['Low'], sigma=1)
        
        return daily_counts
    
    # Calculate counts for each feature
    pitch_counts = calculate_counts('Pitch')
    jitter_counts = calculate_counts('Jitter')
    shimmer_counts = calculate_counts('Shimmer')
    
    return pitch_counts, jitter_counts, shimmer_counts

def calculate_period_statistics(daily_counts):
    """
    Calculate statistics for different periods around the dosing.
    
    Parameters:
    -----------
    daily_counts : pandas.DataFrame
        The dataframe with daily deviation counts
        
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
        period_data = daily_counts[(daily_counts['RelativeDate'] >= start) & 
                          (daily_counts['RelativeDate'] < end)]
        
        period_stats[period_name] = {
            'High': period_data['SmoothedHigh'].mean(),
            'Low': period_data['SmoothedLow'].mean(),
            'Days': end - start
        }
    
    return period_stats, periods

def pairwise_comparisons(periods, anomaly_type, daily_counts):
    """
    Perform pairwise t-tests between periods.
    
    Parameters:
    -----------
    periods : dict
        Dictionary with period definitions
    anomaly_type : str
        Type of anomaly to compare ('High' or 'Low')
    daily_counts : pandas.DataFrame
        The dataframe with daily deviation counts
        
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
            period1_data = daily_counts[(daily_counts['RelativeDate'] >= start1) & 
                                       (daily_counts['RelativeDate'] < end1)][f'Smoothed{anomaly_type}']
            
            # Get data for period 2
            start2, end2 = periods[period2]
            period2_data = daily_counts[(daily_counts['RelativeDate'] >= start2) & 
                                       (daily_counts['RelativeDate'] < end2)][f'Smoothed{anomaly_type}']
            
            # Use independent t-test instead of paired t-test since periods may have different lengths
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
    
    # Apply FDR correction
    p_values = [comp['p-value'] for comp in comparisons]
    rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Add corrected p-values to comparisons
    for i, comp in enumerate(comparisons):
        comp['p-value-fdr'] = p_corrected[i]
        # Print the p-values
        print(f"Comparison between {comp['Period 1']} and {comp['Period 2']}: p-value = {comp['p-value']}, p-value (FDR) = {comp['p-value-fdr']}")
    
    return comparisons

def print_detailed_statistics(comparisons, feature_name):
    """
    Print detailed statistical test results in a formatted manner.
    
    Parameters:
    -----------
    comparisons : list
        List of dictionaries with comparison results
    feature_name : str
        Name of the feature being compared (e.g., 'PITCH', 'JITTER', 'SHIMMER')
    """
    print(f"\n--- {feature_name.upper()} Comparisons ---\n")
    
    for deviation_type in ['High', 'Low']:
        print(f"{deviation_type} Deviations from Mean:")
        for comp in comparisons[deviation_type]:
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

def plot_bar_comparison(ax, stats, title, counts, periods):
    """
    Plot bar comparison with significance indicators.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    stats : dict
        Dictionary with statistics for each period
    title : str
        Title of the plot
    counts : pandas.DataFrame
        The dataframe with daily counts
    periods : dict
        Dictionary with period definitions
    """
    x = np.arange(len(periods))
    width = 0.35
    
    high_means = [stat_dict['High'] for stat_dict in stats.values()]
    low_means = [stat_dict['Low'] for stat_dict in stats.values()]
    
    # Add darker edge color for better visibility
    bars1 = ax.bar(x - width/2, high_means, width, label='High Deviations from Mean', 
                  color='red', alpha=0.7, edgecolor='darkred')
    bars2 = ax.bar(x + width/2, low_means, width, label='Low Deviations from Mean', 
                  color='blue', alpha=0.7, edgecolor='darkblue')
    
    # Add error bars
    def add_error_bars(counts, period_ranges, anomaly_type, x_positions):
        errors = []
        for start, end in period_ranges:
            period_data = counts[
                (counts['RelativeDate'] >= start) & 
                (counts['RelativeDate'] < end)
            ][f'Smoothed{anomaly_type}']
            # Calculate standard error (std / sqrt(n))
            errors.append(period_data.std() / np.sqrt(len(period_data)))
        
        ax.errorbar(x_positions, 
                   [stat_dict[anomaly_type] for stat_dict in stats.values()],
                   yerr=errors,
                   fmt='none', color='black', capsize=5)
    
    add_error_bars(counts, [p for p in periods.values()], 'High', x - width/2)
    add_error_bars(counts, [p for p in periods.values()], 'Low', x + width/2)
    
    # Calculate pairwise comparisons for significance testing
    high_comparisons = pairwise_comparisons(periods, 'High', counts)
    low_comparisons = pairwise_comparisons(periods, 'Low', counts)
    
    # Add significance bars for high deviations (red)
    y_max = max(max(high_means), max(low_means)) * 1.1
    bar_height = y_max * 0.1
    
    # Add significance bars for high deviations
    for comp in high_comparisons:
        if comp['p-value-fdr'] < 0.05:  # Only show significant comparisons
            idx1 = list(periods.keys()).index(comp['Period 1'])
            idx2 = list(periods.keys()).index(comp['Period 2'])
            
            # Draw the line
            line_y = y_max + bar_height
            ax.plot([x[idx1] - width/2, x[idx2] - width/2], [line_y, line_y], 
                   color='red', linewidth=1.5)
            
            # Add significance stars - positioned closer to the line
            if comp['p-value-fdr'] < 0.001:
                stars = '***'
            elif comp['p-value-fdr'] < 0.01:
                stars = '**'
            else:
                stars = '*'
                
            # Position the stars just above the line (reduced vertical offset)
            ax.text((x[idx1] + x[idx2])/2 - width/2, line_y - 0.1, 
                   stars, ha='center', va='bottom', color='red')
            
            y_max += bar_height * 0.5  # Increment for next bar
    
    # Add significance bars for low deviations (blue)
    y_max += bar_height  # Add extra space between high and low significance bars
    
    for comp in low_comparisons:
        if comp['p-value-fdr'] < 0.05:  # Only show significant comparisons
            idx1 = list(periods.keys()).index(comp['Period 1'])
            idx2 = list(periods.keys()).index(comp['Period 2'])
            
            # Draw the line
            line_y = y_max + bar_height
            ax.plot([x[idx1] + width/2, x[idx2] + width/2], [line_y, line_y], 
                   color='blue', linewidth=1.5)
            
            # Add significance stars - positioned closer to the line
            if comp['p-value-fdr'] < 0.001:
                stars = '***'
            elif comp['p-value-fdr'] < 0.01:
                stars = '**'
            else:
                stars = '*'
                
            # Position the stars just above the line (reduced vertical offset)
            ax.text((x[idx1] + x[idx2])/2 + width/2, line_y - 0.1, 
                   stars, ha='center', va='bottom', color='blue')
            
            y_max += bar_height * 0.5  # Increment for next bar
    
    # Set y-axis limit to accommodate significance bars
    ax.set_ylim(0, 6)
    
    # Finalize plot
    ax.set_title(title)
    ax.set_ylabel('Mean Frequency of Deviations from Mean')
    ax.set_xticks(x)
    ax.set_xticklabels(list(periods.keys()), rotation=45, ha='right')
    ax.legend()

def plot_time_series(ax, counts, title, periods, period_colors):
    """
    Plot time series of deviations with proper period shading.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    counts : pandas.DataFrame
        The dataframe with daily counts
    title : str
        Title of the plot
    periods : dict
        Dictionary with period definitions
    period_colors : dict
        Dictionary with colors for each period
    """
    ax.plot(counts['RelativeDate'], counts['SmoothedHigh'], 
            color='red', label='High Deviations from Mean', linewidth=2)
    ax.plot(counts['RelativeDate'], counts['SmoothedLow'], 
            color='blue', label='Low Deviations from Mean', linewidth=2)
    
    # Add vertical line at day 0 (dosing)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Add period shading with distinct colors for each period
    legend_handles = []
    legend_labels = []
    
    # Add the line plots to legend
    legend_handles.extend([
        plt.Line2D([0], [0], color='red', lw=2),
        plt.Line2D([0], [0], color='blue', lw=2)
    ])
    legend_labels.extend(['High Deviations from Mean', 'Low Deviations from Mean'])
    
    # Add period shading with distinct colors - ensure exact match with periods
    for period_name, (start, end) in periods.items():
        span = ax.axvspan(start, end, alpha=0.3, 
                         color=period_colors[period_name])
        
        # Add to legend
        legend_handles.append(span)
        legend_labels.append(period_name)
    
    # Add labels
    ax.set_xlabel('Days Relative to Dosing')
    ax.set_ylabel('Frequency of Deviations from Mean')
    ax.set_title(title)
    
    # Add legend
    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # Find and mark peak days
    high_peak_date = counts.loc[counts['SmoothedHigh'].idxmax(), 'RelativeDate']
    high_peak_value = counts['SmoothedHigh'].max()
    
    low_peak_date = counts.loc[counts['SmoothedLow'].idxmax(), 'RelativeDate']
    low_peak_value = counts['SmoothedLow'].max()
    
    # Add peak highlights
    ax.scatter(high_peak_date, high_peak_value, 
              color='red', s=100, zorder=5)
    ax.scatter(low_peak_date, low_peak_value, 
              color='blue', s=100, zorder=5)

def plot_multipanel_vocal_dynamics(df, pitch_counts, jitter_counts, shimmer_counts, 
                                  pitch_stats, jitter_stats, shimmer_stats, periods):
    """
    Create a multipanel figure with pitch, jitter, and shimmer analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with vocal features and deviations
    pitch_counts, jitter_counts, shimmer_counts : pandas.DataFrame
        The dataframes with daily deviation counts
    pitch_stats, jitter_stats, shimmer_stats : dict
        Dictionaries with period statistics for each feature
    periods : dict
        Dictionary with period definitions
        
    Returns:
    --------
    tuple
        Tuple containing the figure and axes
    """
    # Create figure with 2x3 subplots and extra space on the right for legends
    fig = plt.figure(figsize=(20, 10))
    
    # Create a gridspec with 2 rows and 4 columns (last column for legends)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.25])
    
    # Create axes for the plots
    axes = []
    for row in range(2):
        row_axes = []
        for col in range(3):
            row_axes.append(fig.add_subplot(gs[row, col]))
        axes.append(row_axes)
    
    # Create axes for the legends - position them a bit to the left
    legend_ax_top = fig.add_subplot(gs[0, 3])
    legend_ax_bottom = fig.add_subplot(gs[1, 3])
    
    # Turn off axes for legend subplots
    legend_ax_top.axis('off')
    legend_ax_bottom.axis('off')
    
    # Define distinct colors for each weekly period
    period_colors = {
        'Week -2 (-14 to -7)': 'lightgray',
        'Week -1 (-7 to 0)': 'lightgreen',
        'Week +1 (0 to 7)': 'paleturquoise',
        'Week +2 (7 to 14)': 'lavender'
    }
    
    # Modified plot_time_series function that doesn't add a legend
    def plot_time_series_no_legend(ax, counts, title, periods, period_colors):
        ax.plot(counts['RelativeDate'], counts['SmoothedHigh'], 
                color='red', label='High Deviations from Mean', linewidth=2)
        ax.plot(counts['RelativeDate'], counts['SmoothedLow'], 
                color='blue', label='Low Deviations from Mean', linewidth=2)
        
        # Add vertical line at day 0 (dosing)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add period shading with distinct colors - ensure exact match with periods
        for period_name, (start, end) in periods.items():
            ax.axvspan(start, end, alpha=0.3, 
                      color=period_colors[period_name])
        
        # Add labels
        ax.set_xlabel('Days Relative to Dosing')
        ax.set_ylabel('Frequency of Deviations from Mean')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Find and mark peak days
        high_peak_date = counts.loc[counts['SmoothedHigh'].idxmax(), 'RelativeDate']
        high_peak_value = counts['SmoothedHigh'].max()
        
        low_peak_date = counts.loc[counts['SmoothedLow'].idxmax(), 'RelativeDate']
        low_peak_value = counts['SmoothedLow'].max()
        
        # Add peak highlights
        ax.scatter(high_peak_date, high_peak_value, 
                  color='red', s=100, zorder=5)
        ax.scatter(low_peak_date, low_peak_value, 
                  color='blue', s=100, zorder=5)
    
    # Modified plot_bar_comparison function that doesn't add a legend
    def plot_bar_comparison_no_legend(ax, stats, title, counts, periods):
        x = np.arange(len(periods))
        width = 0.35
        
        high_means = [stat_dict['High'] for stat_dict in stats.values()]
        low_means = [stat_dict['Low'] for stat_dict in stats.values()]
        
        # Add darker edge color for better visibility
        bars1 = ax.bar(x - width/2, high_means, width, label='High Deviations from Mean', 
                      color='red', alpha=0.7, edgecolor='darkred')
        bars2 = ax.bar(x + width/2, low_means, width, label='Low Deviations from Mean', 
                      color='blue', alpha=0.7, edgecolor='darkblue')
        
        # Add error bars
        def add_error_bars(counts, period_ranges, anomaly_type, x_positions):
            errors = []
            for start, end in period_ranges:
                period_data = counts[
                    (counts['RelativeDate'] >= start) & 
                    (counts['RelativeDate'] < end)
                ][f'Smoothed{anomaly_type}']
                # Calculate standard error (std / sqrt(n))
                errors.append(period_data.std() / np.sqrt(len(period_data)))
        
        add_error_bars(counts, [p for p in periods.values()], 'High', x - width/2)
        add_error_bars(counts, [p for p in periods.values()], 'Low', x + width/2)
        
        # Calculate pairwise comparisons for significance testing
        high_comparisons = pairwise_comparisons(periods, 'High', counts)
        low_comparisons = pairwise_comparisons(periods, 'Low', counts)
        
        # Start significance lines at a consistent height relative to the maximum bar
        max_bar_height = max(max(high_means), max(low_means))
        y_start = max_bar_height * 1.2
        line_spacing = 0.4  # Consistent spacing between lines
        
        # Add significance bars for high deviations (red)
        y_pos = y_start
        
        # Add significance bars for high deviations
        for comp in high_comparisons:
            if comp['p-value-fdr'] < 0.05:  # Only show significant comparisons
                idx1 = list(periods.keys()).index(comp['Period 1'])
                idx2 = list(periods.keys()).index(comp['Period 2'])
                
                # Draw the line
                ax.plot([x[idx1] - width/2, x[idx2] - width/2], [y_pos, y_pos], 
                       color='red', linewidth=1.5)
                
                # Add significance stars - positioned closer to the line
                if comp['p-value-fdr'] < 0.001:
                    stars = '***'
                elif comp['p-value-fdr'] < 0.01:
                    stars = '**'
                else:
                    stars = '*'
                    
                # Position the stars just above the line (reduced vertical offset)
                ax.text((x[idx1] + x[idx2])/2 - width/2, y_pos - 0.1, 
                       stars, ha='center', va='bottom', color='red')
                
                y_pos += line_spacing  # Use consistent spacing
        
        # Add significance bars for low deviations (blue)
        # Start at a consistent height above the high deviation lines
        y_pos = y_start + (len([c for c in high_comparisons if c['p-value-fdr'] < 0.05]) + 1) * line_spacing
        
        for comp in low_comparisons:
            if comp['p-value-fdr'] < 0.05:  # Only show significant comparisons
                idx1 = list(periods.keys()).index(comp['Period 1'])
                idx2 = list(periods.keys()).index(comp['Period 2'])
                
                # Draw the line
                ax.plot([x[idx1] + width/2, x[idx2] + width/2], [y_pos, y_pos], 
                       color='blue', linewidth=1.5)
                
                # Add significance stars - positioned closer to the line
                if comp['p-value-fdr'] < 0.001:
                    stars = '***'
                elif comp['p-value-fdr'] < 0.01:
                    stars = '**'
                else:
                    stars = '*'
                    
                # Position the stars just above the line (reduced vertical offset)
                ax.text((x[idx1] + x[idx2])/2 + width/2, y_pos - 0.1, 
                       stars, ha='center', va='bottom', color='blue')
                
                y_pos += line_spacing  # Use consistent spacing
        
        # Set y-axis limit to accommodate significance bars with some padding
        max_y_pos = y_start + (len([c for c in high_comparisons if c['p-value-fdr'] < 0.05]) + 
                          len([c for c in low_comparisons if c['p-value-fdr'] < 0.05]) + 2) * line_spacing
        
        # Set consistent y-axis limits for all plots
        ax.set_ylim(0, max(6, max_y_pos))
        
        # Finalize plot
        ax.set_title(title)
        ax.set_ylabel('Mean Frequency of Deviations from Mean')
        ax.set_xticks(x)
        ax.set_xticklabels(list(periods.keys()), rotation=45, ha='right')
    
    # Plot time series with consistent period shading
    plot_time_series_no_legend(axes[0][0], pitch_counts, 'Pitch Deviations from Mean Around Dosing', periods, period_colors)
    plot_time_series_no_legend(axes[0][1], jitter_counts, 'Jitter Deviations from Mean Around Dosing', periods, period_colors)
    plot_time_series_no_legend(axes[0][2], shimmer_counts, 'Shimmer Deviations from Mean Around Dosing', periods, period_colors)
    
    # Plot bar comparisons with significance indicators
    plot_bar_comparison_no_legend(axes[1][0], pitch_stats, 'Period Comparison of Pitch Variability', pitch_counts, periods)
    plot_bar_comparison_no_legend(axes[1][1], jitter_stats, 'Period Comparison of Jitter Variability', jitter_counts, periods)
    plot_bar_comparison_no_legend(axes[1][2], shimmer_stats, 'Period Comparison of Shimmer Variability', shimmer_counts, periods)
    
    # Create legend for top row (time series)
    # Create handles for the legend
    line_handles = [
        plt.Line2D([0], [0], color='red', lw=2, label='High Deviations from Mean'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Low Deviations from Mean')
    ]
    
    # Add period patches to the legend
    patch_handles = []
    for period_name, color in period_colors.items():
        patch_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.3, label=period_name))
    
    # Add the legend to the top row - moved slightly to the left
    legend_ax_top.legend(handles=line_handles + patch_handles, loc='center', fontsize='medium')
    
    # Create legend for bottom row (bar charts)
    # Create handles for the legend
    bar_handles = [
        plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.7, edgecolor='darkred', label='High Deviations from Mean'),
        plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.7, edgecolor='darkblue', label='Low Deviations from Mean')
    ]
    
    # Add significance indicators to the legend
    sig_handles = [
        plt.Line2D([0], [0], color='black', lw=0, label='Significance:'),
        plt.Line2D([0], [0], color='black', lw=0, label='* p < 0.05'),
        plt.Line2D([0], [0], color='black', lw=0, label='** p < 0.01'),
        plt.Line2D([0], [0], color='black', lw=0, label='*** p < 0.001')
    ]
    
    # Add the legend to the bottom row - moved slightly to the left
    legend_ax_bottom.legend(handles=bar_handles + sig_handles, loc='center', fontsize='medium')
    
    plt.tight_layout()
    fig.suptitle('Temporal Evolution and Period Comparison of Vocal Deviations from Mean Relative to 5-MeO-DMT Dosing', fontsize=16, y=1.02)
    
    return fig, axes

def add_period_column(df):
    """
    Add a period column to the dataframe based on the relative date.
    0: -14 to -7
    1: -7 to 0
    2: 0 to 7
    3: 7 to 14
    """
    df['Period'] = pd.cut(df['RelativeDate'], bins=[-14, -7, 0, 7, 14], labels=[0, 1, 2, 3])
    return df

def analyze_vocal_dynamics(vocal_features_path, output_path=None):
    """
    Perform complete vocal dynamics analysis.
    
    Parameters:
    -----------
    vocal_features_path : str
        Path to the vocal features CSV file
    output_path : str, optional
        Path to save the output figure
        
    Returns:
    --------
    tuple
        Tuple containing the dataframe with deviations, test results, and questionnaire prediction results
    """
    print("\n=== RUNNING VOCAL DYNAMICS ANALYSIS ===\n")
    
    # Load data
    df = load_data(vocal_features_path)
    print("Data loaded successfully.")
    
    # Filter valid users
    df = filter_valid_users(df)
    print(f"Filtered valid users: {df['UserID'].nunique()} users remaining.")

    # Add period column
    df = add_period_column(df)
    print("Period column added.")
    
    # Identify vocal deviations
    df = identify_vocal_deviations(df)
    print("Vocal deviations identified.")
    
    # Calculate daily deviation counts
    pitch_counts, jitter_counts, shimmer_counts = calculate_daily_deviation_counts(df)
    print("Daily deviation counts calculated.")
    
    # Calculate period statistics
    pitch_stats, periods = calculate_period_statistics(pitch_counts)
    jitter_stats, _ = calculate_period_statistics(jitter_counts)
    shimmer_stats, _ = calculate_period_statistics(shimmer_counts)
    print("Period statistics calculated.")
    
    # Perform pairwise comparisons
    pitch_comparisons = {
        'High': pairwise_comparisons(periods, 'High', pitch_counts),
        'Low': pairwise_comparisons(periods, 'Low', pitch_counts)
    }
    jitter_comparisons = {
        'High': pairwise_comparisons(periods, 'High', jitter_counts),
        'Low': pairwise_comparisons(periods, 'Low', jitter_counts)
    }
    shimmer_comparisons = {
        'High': pairwise_comparisons(periods, 'High', shimmer_counts),
        'Low': pairwise_comparisons(periods, 'Low', shimmer_counts)
    }
    
    # Print detailed statistics
    print("\n=== DETAILED STATISTICAL TEST RESULTS ===\n")
    print_detailed_statistics(pitch_comparisons, 'PITCH')
    print_detailed_statistics(jitter_comparisons, 'JITTER')
    print_detailed_statistics(shimmer_comparisons, 'SHIMMER')
    
    # Plot results
    fig, axes = plot_multipanel_vocal_dynamics(df, pitch_counts, jitter_counts, shimmer_counts, 
                                               pitch_stats, jitter_stats, shimmer_stats, periods)
    print("Plots generated.")
    
    # Save the figure if output path is provided
    if output_path:
        fig.savefig(output_path)
        print(f"Figure saved to {output_path}")
    
    return df, (pitch_counts, jitter_counts, shimmer_counts), (pitch_stats, jitter_stats, shimmer_stats)

if __name__ == "__main__":
    # Example usage
    vocal_features_path = "/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/vocal_features.csv"
    
    # Run analysis
    df, anomaly_counts, test_results = analyze_vocal_dynamics(
        vocal_features_path,
        output_path="vocal_dynamics_results.png"
    )

    df.to_csv("/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/vocal_features_with_deviations.csv", index=False)
