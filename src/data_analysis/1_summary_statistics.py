import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def summary_plot(df, export_path):

    # Set the figure size
    plt.figure(figsize=(11, 5))

    # Create a new DataFrame to count unique Recording IDs (participants) per RelativeDate
    unique_id_count = df.groupby('RelativeDate')['RecordingID'].nunique().reset_index()
    unique_recording_id_count = df['RecordingID'].nunique()
    print(unique_recording_id_count)

    # Plot: Use plt.bar to create a bar chart with continuous x-axis
    plt.subplot(1, 2, 1)  # Two rows, one column, second plot
    plt.bar(unique_id_count['RelativeDate'], unique_id_count['RecordingID'], color='slateblue', alpha=0.7)
    plt.title('Number of Unique Entries Submitted \nby Day Relative to 5-MeO-DMT Session\n')
    plt.xlabel('Days Before / After 5-MeO-DMT Dosing (Day 0)')
    plt.ylabel('Number of Unique Voice Notes Submitted')

    # Set the x-ticks for the second plot to match the first plot
    plt.xticks(np.arange(-20, 21, 5))

    plt.axvspan(-7, 7, color='blue', alpha=0.1, label="Day Range: -7 to +7")
    plt.axvspan(-14, 14, color='black', alpha=0.1, label="Day Range: -14 to +14")
    # add box with total 
    plt.text(0.98, 0.95, 
            f'Total: {unique_recording_id_count}', 
            horizontalalignment='right', 
            verticalalignment='top', 
            transform=plt.gca().transAxes, 
            fontsize=12, 
            fontweight='bold', 
            color='black')

    plt.text(0.98, 0.88, 
            f'Included: {df[(df["RelativeDate"] >= -14) & (df["RelativeDate"] <= 14)]["RecordingID"].nunique()}', 
            horizontalalignment='right', 
            verticalalignment='top', 
            transform=plt.gca().transAxes, 
            fontsize=10, 
            color='black')


    # Plot: Histogram of the number of sentences submitted per day relative to the retreat
    plt.subplot(1, 2, 2)  # Two rows, one column, first plot
    sns.histplot(df['RelativeDate'], bins=10, kde=True, color='cornflowerblue')
    plt.title('Distribution of Sentences Submitted \nby Day Relative to 5-MeO-DMT Session\n')
    plt.xlabel('Days Before / After 5-MeO-DMT Dosing (Day 0)')
    plt.ylabel('Number of Sentences Submitted')
    plt.axvspan(-7, 7, color='blue', alpha=0.1, label="Active Notification Period (-7 to +7)")
    plt.axvspan(-14, 14, color='black', alpha=0.1, label="Included in Analysis (-14 to +14)")
    # add box with total 
    plt.text(0.98, 0.95, 
            f'Total: {df.shape[0]}', 
            horizontalalignment='right', 
            verticalalignment='top', 
            transform=plt.gca().transAxes, 
            fontsize=12, 
            fontweight='bold', 
            color='black')

    plt.text(0.98, 0.88, 
            f'Included: {df[(df["RelativeDate"] >= -14) & (df["RelativeDate"] <= 14)].shape[0]}', 
            horizontalalignment='right', 
            verticalalignment='top', 
            transform=plt.gca().transAxes, 
            fontsize=10, 
            color='black')


    # Add legend on the right-hand sides
    plt.legend(loc='upper center', bbox_to_anchor=(0.4, -0.2), ncol=3, fontsize=10)


    # plt.suptitle('Daily Submission Patterns and Participant Engagement Before and After 5-MeO-DMT Dosing', y=1, fontsize=16, fontweight='bold')
    plt.tight_layout()  # Adjust layout to prevent overlap

    # export 
    plt.savefig(export_path, dpi=300)

    plt.show()

if __name__ == '__main__':  

    CORE_DIR = '/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/'

    df = pd.read_csv(f'{CORE_DIR}/data/final/sentence_level.csv')
    # for this plot, all RelativeDate equal to -0.5 or 0.5 should be changed to 0 
    df['RelativeDate'] = df['RelativeDate'].apply(lambda x: 0 if x == -0.5 or x == 0.5 else x)
    summary_plot(df, f'{CORE_DIR}/outputs/figures/summary_statistics.png')
   
    mean_recordings_per_user = df.groupby('UserID')['RecordingID'].nunique().mean()
    std_recordings_per_user = df.groupby('UserID')['RecordingID'].nunique().std()
    print(f'Mean recordings per user: {mean_recordings_per_user:.1f} +/- {std_recordings_per_user:.1f}')

    mean_sentences_per_user = df.groupby('UserID')['RecordingID'].count().mean()
    std_sentences_per_user = df.groupby('UserID')['RecordingID'].count().std()
    print(f'Mean sentences per user: {mean_sentences_per_user:.1f} +/- {std_sentences_per_user:.1f}')


