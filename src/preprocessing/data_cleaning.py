"""
Data Cleaning Pipeline for Retreat Journal Entries
===============================================

This script processes and cleans journal entries from retreat participants, handling multiple iterations
of manual cleaning and implementing data anonymization. The pipeline includes:

1. Loading and cleaning retreat dates and participant entries
2. Timezone correction (UTC to America/Mexico_City)
3. Processing manually cleaned transcripts (two iterations)
4. Anonymizing sensitive information in transcripts
5. Maintaining consistent column ordering throughout processing

Key Assumptions:
---------------
- Retreat dates are in dd-mm-yyyy format
- All timestamps are initially in UTC
- All retreats take place in Mexico City timezone (UTC-6)
- Retreats start at 8:00 AM local time
- Input files:
    * retreat_dates.csv: Contains UserID and StartDate
    * retreat_entries.csv: Original entries with UserID, RecordingID, Timestamp, Transcript
    * cleaned_transcripts_iteration1.csv: First round of manual cleaning
    * cleaned_transcripts_iteration2.csv: Second round of manual cleaning

Column Order:
------------
The script maintains a consistent column order (if columns exist):
- UserID: Unique identifier for each participant
- RecordingID: Unique identifier for each entry
- Timestamp: Original UTC timestamp
- TimestampCorrected: Local time (Mexico City)
- RetreatDate: Start date of retreat
- RelativeDate: Days relative to retreat start
- TranscriptOriginal: Original uncleaned text
- TranscriptClean: Manually cleaned text
- AnonymisedTranscript: Cleaned text with sensitive info removed
- ManualClean: Boolean indicating manual cleaning status

Output Files:
------------
- journals_v0.csv: Initial processing result
- journals_v1.csv: After second iteration cleaning
- journals_v2.csv: Final anonymized version

Dependencies:
------------
- pandas: Data processing
- anonymise.TextAnonymizer: Custom anonymization tool
"""

import pandas as pd
import numpy as np
from .anonymise import TextAnonymizer
import re

FINAL_COLORDER = ['UserID', 'RecordingID', 'Timestamp', 'TimestampCorrected', 'RetreatDate', 
                  'RelativeDate', 'PrePost','TranscriptOriginal', 'TranscriptClean', 'AnonymisedTranscript', 'ManualClean']

def load_and_clean_retreat_dates(file_path):
    """Load and clean retreat dates from CSV file.
    
    Args:
        file_path (str): Path to retreat dates CSV file
        
    Returns:
        pd.DataFrame: Cleaned retreat dates dataframe
    """
    retreat_dates = pd.read_csv(file_path)
    retreat_dates['StartDate'] = pd.to_datetime(retreat_dates['StartDate'], format='%d-%m-%Y')
    # add 1 day to StartDate as 1st day is prep and 2nd day of retreat is dosing
    retreat_dates['StartDate'] = retreat_dates['StartDate'] + pd.Timedelta(days=1)

    print("✔︎ Retreat dates loaded and cleaned.")

    return retreat_dates.rename(columns={'StartDate': 'RetreatDate'})

def load_and_clean_entries(entries_path, cleaned_entries_path):
    """Load and merge original entries with manually cleaned entries.
    
    Args:
        entries_path (str): Path to original entries CSV
        cleaned_entries_path (str): Path to manually cleaned entries CSV
        
    Returns:
        pd.DataFrame: Merged and cleaned entries dataframe
    """
    # Load raw entries
    entries = pd.read_csv(entries_path)
    entries['Timestamp'] = pd.to_datetime(entries['Timestamp'], unit='ms')
    
    # Load manually cleaned entries
    cleaned_entries = pd.read_csv(cleaned_entries_path)
    
    # Merge and track cleaned entries
    entries = entries.merge(
        cleaned_entries[['RecordingID', 'Transcript']], 
        on='RecordingID', 
        how='left', 
        suffixes=('', 'Clean')
    )
    
    # Add tracking column and rename for clarity
    entries = entries.rename(columns={'Transcript': 'TranscriptOriginal'})
    entries['ManualClean'] = entries['TranscriptClean'].notna()
    
    # Replace the desired_order with FINAL_COLORDER
    existing_cols = [col for col in FINAL_COLORDER if col in entries.columns]
    remaining_cols = [col for col in entries.columns if col not in FINAL_COLORDER]

    print("✔︎ First iteration cleaning complete.")
    
    return entries[existing_cols + remaining_cols]

def process_second_iteration(entries_path):
    """Process second iteration of manually cleaned entries.
    
    Args:
        entries_path (str): Path to second iteration of cleaned entries
        
    Returns:
        pd.DataFrame: Processed entries with updated ManualClean status
    """
    cleaned_entries = pd.read_csv(entries_path)
    
    # Update ManualClean status
    for index, row in cleaned_entries.iterrows():
        if not row['ManualClean'] and not pd.isna(row['TranscriptClean']):
            cleaned_entries.at[index, 'ManualClean'] = True
    
    # Find and remove entries where TranscriptClean only contains dates
    date_pattern = r'^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}$'
    date_only_mask = cleaned_entries['TranscriptClean'].str.match(date_pattern, na=False)
    
    if date_only_mask.any():
        print(f"Removing {date_only_mask.sum()} entries containing only dates")
        cleaned_entries = cleaned_entries[~date_only_mask]
    
    # remove all rows where TranscriptClean is ""
    cleaned_entries = cleaned_entries[cleaned_entries['TranscriptClean'] != ""]
    # remove all rows where length of string in TranscriptClean is < 1
    cleaned_entries = cleaned_entries[cleaned_entries['TranscriptClean'].str.len() > 1]
    
    # Filter to only manually cleaned entries
    cleaned_entries = cleaned_entries[cleaned_entries['ManualClean']]

    # drop ManualClean column
    cleaned_entries = cleaned_entries.drop(columns=['ManualClean'])

    print("✔︎ Second iteration cleaning complete.")
    
    return cleaned_entries

def anonymize_transcripts(df):
    """Anonymize the cleaned transcripts using NER.
    
    Args:
        df (pd.DataFrame): DataFrame containing TranscriptClean column
        
    Returns:
        pd.DataFrame: DataFrame with new AnonymisedTranscript column
    """
    # Initialize anonymizer with default model
    anonymizer = TextAnonymizer()
    
    # Add common words to ignore
    ignore_words = []
    anonymizer.add_to_ignore_list(ignore_words)
    
    # Process only rows that have cleaned transcripts
    mask = df['TranscriptClean'].notna()
    df.loc[mask, 'AnonymisedTranscript'] = df.loc[mask, 'TranscriptClean'].apply(anonymizer.anonymize_text)
    
    print("✔︎ Anonymisation complete.")
    
    return df

def correct_timestamp_to_timezone(df, timezone='America/Mexico_City', retreat_start_hour=8):
    """Correct the timestamp to the timezone of the retreat.
    
    Args:
        df (pd.DataFrame): DataFrame containing Timestamp and RetreatDate columns
        timezone (str): Timezone name (default: 'America/Mexico_City')
        retreat_start_hour (int): Hour of day when retreats start (default: 8)
        
    Returns:
        pd.DataFrame: DataFrame with corrected Timestamp and RetreatDate columns
    """
    if 'Timestamp' not in df.columns:
        raise ValueError("DataFrame must contain 'Timestamp' column")
        
    try:
        df['TimestampCorrected'] = (df['Timestamp']
            .dt.tz_localize('UTC')
            .dt.tz_convert(timezone)
            .dt.tz_localize(None)
        )
        
        # Parse RetreatDate 
        df['RetreatDate'] = pd.to_datetime(df['RetreatDate'])
        df['RetreatDate'] = df['RetreatDate'] + pd.Timedelta(hours=retreat_start_hour)
        
        df['RelativeDate'] = (
            df['TimestampCorrected'] - df['RetreatDate']
        ).dt.days
        
        df['PrePost'] = np.where(df['RelativeDate'] < 0, 0, 1)

        df = manual_correction_of_prepost(df)

        return df
        
    except Exception as e:
        raise ValueError(f"Error processing dates: {str(e)}")

def final_cleaning(df):
    """Perform final cleaning steps on the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # remove all rows where length of string in TranscriptClean is < 1
    
    print(df['UserID'].unique())
    df.loc[df['UserID'] == 'P4O85', 'UserID'] = 'P4085'
    print(df['UserID'].unique())
    print("✔︎ Username for P4085 corrected.")

    # Remove participant with Participant ID = PRO71
    df = df[df['UserID'] != 'PRO71']
    print("✔︎ Participant with Participant ID = PRO71 removed.")

    # sort by RetreatDate, ascending and then UserID, ascending
    df = df.sort_values(by=['RetreatDate', 'UserID'], ascending=True)
    print("✔︎ Dataframe sorted by RetreatDate and UserID.")

    # reset the index
    df = df.reset_index(drop=True)
    print("✔︎ Dataframe index reset.")
    
    return df

def manual_correction_of_prepost(df):
    """Perform manual correction of PrePost column.
    Entries submitted on Day 0 were manually checked to see if they're submitted before or after the retreat started.
    This is based on recording ID. 
    Those that were submitted before the retreat started were assigned a RelativeDate value of -0.5.
    Those that were submitted after the retreat started were assigned a RelativeDate value of 0.5.
    """

    ENTRIES_TO_CORRECT = {
        -0.5: ['GGnB3SMy82', 'uaOZtIQXcA', '3CkMVS61y9', 'MBLYsVX47L', 'piw5ih2tX8', '2h94NhYnIZ'],
        0.5: ['zPasMI8YnS', 'ru5DP32OZL', 'VybMKVp3O6']
    }

    for relative_date, entries in ENTRIES_TO_CORRECT.items():
        df.loc[df['RecordingID'].isin(entries), 'RelativeDate'] = relative_date
        if relative_date == -0.5:
            df.loc[df['RecordingID'].isin(entries), 'PrePost'] = 0
        elif relative_date == 0.5:
            df.loc[df['RecordingID'].isin(entries), 'PrePost'] = 1

    return df


def main():
    # File paths
    BASE_PATH = '/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data'
    RETREAT_DATES_PATH = f'{BASE_PATH}/raw/retreat_dates.csv'
    ENTRIES_PATH = f'{BASE_PATH}/raw/retreat_entries.csv'
    CLEANED_V1_PATH = f'{BASE_PATH}/qual_edits/cleaned_transcripts_iteration1.csv'
    CLEANED_V2_PATH = f'{BASE_PATH}/qual_edits/cleaned_transcripts_iteration2.csv'
    OUTPUT_V0_PATH = f'{BASE_PATH}/interim/journals_v0.csv'
    OUTPUT_V1_PATH = f'{BASE_PATH}/interim/journals_v1.csv'
    FINAL_OUTPUT_PATH = f'{BASE_PATH}/processed/journals_full.csv'

    # Process first iteration
    retreat_dates = load_and_clean_retreat_dates(RETREAT_DATES_PATH)
    entries = load_and_clean_entries(ENTRIES_PATH, CLEANED_V1_PATH)
    
    # Merge with retreat dates and clean
    entries = entries.merge(retreat_dates[['UserID', 'RetreatDate']], 
                          on='UserID', how='left')
    entries = entries[entries['RetreatDate'].notna()].reset_index(drop=True)
    entries = correct_timestamp_to_timezone(entries)
    
    # Save intermediate result
    print("Manual cleaning status:", entries['ManualClean'].value_counts())
    entries.to_csv(OUTPUT_V0_PATH, index=False)
    
    # Process second iteration
    cleaned_v1 = process_second_iteration(CLEANED_V2_PATH)
    
    # Merge with the time-related columns from entries
    time_columns = ['RecordingID', 'Timestamp', 'TimestampCorrected', 'RelativeDate', 'PrePost']
    cleaned_v1 = cleaned_v1.merge(
        entries[time_columns],
        on='RecordingID',
        how='left',
        suffixes=(None, '_drop')  # This will append '_drop' to duplicate columns
    )
    
    # Remove any columns with '_drop' suffix and unwanted columns
    cols_to_drop = [col for col in cleaned_v1.columns if 
                   col.endswith('_drop') or 
                   col not in FINAL_COLORDER]
    cleaned_v1 = cleaned_v1.drop(columns=cols_to_drop)
    
    cleaned_v1.to_csv(OUTPUT_V1_PATH, index=False)

    # Anonymise second iteration
    cleaned_v2 = anonymize_transcripts(cleaned_v1)

    # Only keep columns that are in FINAL_COLORDER
    final_cols = [col for col in FINAL_COLORDER if col in cleaned_v2.columns]
    cleaned_v2 = cleaned_v2[final_cols]
    
    cleaned_v2 = final_cleaning(cleaned_v2)
    print("Final columns in order:", cleaned_v2.columns.tolist())

    cleaned_v2.to_csv(FINAL_OUTPUT_PATH, index=False)
    

    print("✔︎ Final cleaning complete.")

if __name__ == "__main__":
    main()