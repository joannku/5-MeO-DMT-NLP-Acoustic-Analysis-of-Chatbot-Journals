import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import wave


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
    plt.title('Number of Unique Voice Notes Submitted\n', fontsize=14)
    plt.xlabel('Days Before / After 5-MeO-DMT Dosing (Day 0)')
    plt.ylabel('# Voice Notes Submitted')

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
    plt.title('Number of Sentences Submitted\n', fontsize=14)
    plt.xlabel('Days Before / After 5-MeO-DMT Dosing (Day 0)')
    plt.ylabel('# Sentences Submitted')
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


    # Add legend to be centered below both subplots
    handles, labels = plt.gca().get_legend_handles_labels()
    fig = plt.gcf()  # Get current figure
    fig.legend(handles, labels, 
               loc='lower center',  
               bbox_to_anchor=(0.5, -0.05),  # Moved lower by adjusting y-coordinate from -0.05 to -0.15
               ncol=2,  
               fontsize=10, 
               frameon=False)

    # Adjust bottom margin to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Increased bottom margin from 0.15 to 0.2

    # export 
    plt.savefig(export_path, dpi=300, bbox_inches='tight')  # Added bbox_inches='tight' parameter

    plt.show()

if __name__ == '__main__':  

    CORE_DIR = '/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/'

    df = pd.read_csv(f'{CORE_DIR}/data/final/sentence_level.csv')

    # Remove participant with Participant ID = PRO71
    df = df[df['UserID'] != 'PRO71']

    # for this plot, all RelativeDate equal to -0.5 or 0.5 should be changed to 0 
    df['RelativeDate'] = df['RelativeDate'].apply(lambda x: 0 if x == -0.5 or x == 0.5 else x)
    # print count of unique sentences per RelativeDate as a dictionary
    print(df.groupby('RelativeDate')['Sentence'].nunique().to_dict())
    summary_plot(df, f'{CORE_DIR}/outputs/figures/summary_statistics.png')
   
    # Filter data to only include the -14 to +14 day window
    df_window = df[(df['RelativeDate'] >= -14) & (df['RelativeDate'] <= 14)]
    
    # Calculate total duration of WAV files included in the dataset (only for the -14 to +14 day window)
    wav_dir = f'{CORE_DIR}/data/audio/wav'
    total_duration = 0
    file_count = 0
    included_recording_ids = set(df_window['RecordingID'].unique())
    
    # Calculate and display summary statistics for the -14 to +14 day window
    total_sentences = df_window.shape[0]
    total_recordings = len(included_recording_ids)
    total_users = df_window['UserID'].nunique()
    
    # Calculate mean and std of recording durations
    durations = []
    for root, dirs, files in os.walk(wav_dir):
        for file in files:
            if file.endswith('.wav'):
                try:
                    recording_id = file.split('_')[-1].replace('.wav', '')
                    if recording_id in included_recording_ids:
                        file_path = os.path.join(root, file)
                        try:
                            with wave.open(file_path, 'rb') as wav_file:
                                duration = wav_file.getnframes() / wav_file.getframerate()
                                total_duration += duration
                                durations.append(duration)
                                file_count += 1
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")
                except Exception as e:
                    print(f"Error extracting recording ID from {file}: {e}")
    
    mean_duration = np.mean(durations)
    std_duration = np.std(durations)
    
    print("\n----- DATASET SUMMARY (Days -14 to +14) -----")
    print(f"Total unique participants: {total_users}")
    print(f"Total voice recordings: {total_recordings}")
    print(f"Total sentences: {total_sentences}")
    print(f"Average sentences per recording: {total_sentences/total_recordings:.2f}")
    print(f"Total WAV files found: {file_count} (out of {total_recordings} unique recordings in CSV)")
    print(f"Total audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Average duration per recording: {mean_duration:.2f} ± {std_duration:.2f} seconds")
    print(f"Average duration per sentence: {total_duration/total_sentences:.2f} seconds")

    # Continue with the rest of your analysis, but keep using df_window for the -14 to +14 day period
    mean_recordings_per_user = df_window.groupby('UserID')['RecordingID'].nunique().mean()
    std_recordings_per_user = df_window.groupby('UserID')['RecordingID'].nunique().std()
    print(f'Mean recordings per user: {mean_recordings_per_user:.1f} +/- {std_recordings_per_user:.1f}')

    mean_sentences_per_user = df_window.groupby('UserID')['RecordingID'].count().mean()
    std_sentences_per_user = df_window.groupby('UserID')['RecordingID'].count().std()
    print(f'Mean sentences per user: {mean_sentences_per_user:.1f} +/- {std_sentences_per_user:.1f}')

    dfx = pd.read_csv(f'{CORE_DIR}/data/final/means_level+pca.csv')

    # Analysis of participant data
    print("\n----- PARTICIPANT DATA ANALYSIS (Days -14 to +14) -----")
    
    # Identify participants with PRE and POST data within the window
    pre_participants = df_window[df_window['RelativeDate'] < 0]['UserID'].unique()
    post_participants = df_window[df_window['RelativeDate'] > 0]['UserID'].unique()
    
    # Count participants with PRE data
    print(f"Participants with PRE data: {len(pre_participants)}")
    
    # Count participants with POST data
    print(f"Participants with POST data: {len(post_participants)}")
    
    # Count participants with both PRE and POST data (PRE-POST pairs)
    pre_post_pairs = set(pre_participants).intersection(set(post_participants))
    print(f"Participants with PRE-POST pairs: {len(pre_post_pairs)}")

    # Identify participants with at least 2 recordings in PRE and POST periods
    pre_recordings_count = df_window[df_window['RelativeDate'] < 0].groupby('UserID')['RecordingID'].nunique()
    post_recordings_count = df_window[df_window['RelativeDate'] > 0].groupby('UserID')['RecordingID'].nunique()
    
    pre_2plus = pre_recordings_count[pre_recordings_count >= 2].index
    post_2plus = post_recordings_count[post_recordings_count >= 2].index
    
    # Participants with at least 2 recordings both PRE and POST
    pre_post_2plus_pairs = set(pre_2plus).intersection(set(post_2plus))
    print(f"Participants with at least 2 recordings both PRE and POST: {len(pre_post_2plus_pairs)}")

    PSYCHOMETRICS = [
    'survey_aPPS_total',
    'survey_EBI',
    'survey_ASC_OBN',
    'survey_ASC_DED',
    'survey_bSWEBWBS'
]
    
    # Check completion of specific psychometric questionnaires
    # Participants who completed all psychometric questionnaires (no NaN values in the PSYCHOMETRICS columns)
    complete_questionnaires = dfx.dropna(subset=PSYCHOMETRICS)['UserID'].unique() if 'UserID' in dfx.columns else []
    print(f"Participants who completed all psychometric questionnaires: {len(complete_questionnaires)}")
    
    # Participants with PRE + questionnaire data
    pre_with_questionnaire = set(pre_participants).intersection(set(complete_questionnaires))
    print(f"Participants with PRE + psychometric questionnaire data: {len(pre_with_questionnaire)}")
    
    # Participants with at least 2 recordings PRE + questionnaire data
    pre_2plus_with_questionnaire = set(pre_2plus).intersection(set(complete_questionnaires))
    print(f"Participants with at least 2 recordings PRE + psychometric questionnaire data: {len(pre_2plus_with_questionnaire)}")
    
    # Participants with PRE + POST + questionnaire data
    pre_post_with_questionnaire = set(pre_post_pairs).intersection(set(complete_questionnaires))
    print(f"Participants with PRE + POST + psychometric questionnaire data: {len(pre_post_with_questionnaire)}")
    
    # Participants with at least 2 recordings PRE + at least 2 recordings POST + questionnaire data
    pre_post_2plus_with_questionnaire = set(pre_post_2plus_pairs).intersection(set(complete_questionnaires))
    print(f"Participants with at least 2 recordings PRE + at least 2 recordings POST + psychometric questionnaire data: {len(pre_post_2plus_with_questionnaire)}")

