import subprocess
import os
import pandas as pd
import opensmile
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')  # Suppress warnings

def batch_convert_oga_to_wav(directory):
    os.makedirs(os.path.join(directory, "wav"), exist_ok=True)
    for file in os.listdir(directory + "/oga"):
        if file.endswith(".oga"):
            oga_path = os.path.join(directory, "oga", file)
            wav_path = os.path.join(directory, "wav", file.replace(".oga", ".wav"))
            try:
                subprocess.run(["ffmpeg", "-i", oga_path, wav_path], check=True)
                print(f"Converted {file} to WAV format.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {file}: {e}")

def load_and_prepare_data(directory_path):
    """
    Load and prepare audio file paths from a directory.

    Args:
        directory_path (str): Path to the directory containing audio files.

    Returns:
        list: List of paths to audio files.
    """
    # List all files in the directory
    audio_file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.wav')]
    return audio_file_paths

def extract_metadata(file_paths):
    """
    Extract metadata from file paths and add them as columns to a DataFrame.

    Args:
        file_paths (list): List of file paths to extract metadata from.

    Returns:
        pd.DataFrame: DataFrame with metadata columns added.
    """
    # Initialize a list to store metadata
    metadata = []

    for file_path in file_paths:
        # Extract the file name from the path
        file_name = os.path.basename(file_path)
        
        # Split the file name by underscores
        parts = file_name.split('_')
        
        # Extract the relevant parts
        user_id = parts[0]
        recording_id = parts[3].split('.')[0]
        
        # Append the metadata to the list
        metadata.append({
            'file_path': file_path,
            'UserID': user_id,
            'RecordingID': recording_id
        })

    # Convert the list of metadata to a DataFrame
    metadata_df = pd.DataFrame(metadata)
    return metadata_df

def extract_opensmile_features(audio_file_paths):
    """
    Extract vocal features using OpenSMILE.

    Args:
        audio_file_paths (list): List of paths to audio files.

    Returns:
        pd.DataFrame: DataFrame containing vocal features for each audio file.
    """
    # Initialize OpenSMILE with the eGeMAPSv02 feature set
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # Initialize a list to store results
    results = []

    # Process each audio file with a progress bar
    for file_path in tqdm(audio_file_paths, desc="Extracting Features", unit="file"):
        try:
            # Extract features using OpenSMILE
            features = smile.process_file(file_path)
            features['file_path'] = file_path
            results.append(features)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Concatenate all results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)
    return results_df

def export_features_to_csv(results_df, file_path='data/analysis/vocal_features.csv'):
    """
    Export the vocal features DataFrame to a CSV file.

    Args:
        results_df (pd.DataFrame): The DataFrame containing the vocal features.
        file_path (str): The file path where the CSV will be saved.
    """
    results_df.to_csv(file_path, index=False)
    print(f"Completed export of vocal features to {file_path}")

def final_table_format(df, CORE_DIRECTORY):
    """
    Format the final table.
    """

    # In order to not repeat the logic of the data cleaning, we load the already cleaned dataframe and match on RecordingID
    df_journals = pd.read_csv(f'{CORE_DIRECTORY}/data/processed/journals_full.csv')

    # add columns PrePost and RelativeDate and RetreatDate to df
    df = df.merge(df_journals[['RecordingID','Timestamp','TimestampCorrected','RetreatDate','RelativeDate','PrePost']], on='RecordingID', how='left')

    # file_path,UserID,Timestamp,RecordingID and then all other cols
    FIRST_COLS = ['UserID', 'RecordingID','Timestamp','TimestampCorrected','RetreatDate','RelativeDate','PrePost']
    df = df[FIRST_COLS + [col for col in df.columns if col not in FIRST_COLS]]

    # remove rows where Timestamp is None
    df = df[df['Timestamp'].notna()]

    return df

def main(core_directory, audio_directory):
    """
    Main function to load data, extract features, and export results.
    """
    output_file_path = f'{core_directory}/data/processed/vocal_features.csv'

    # check if files were converted to wav by looking at number of files in wav directory
    if len(os.listdir(audio_directory + '/wav')) != len(os.listdir(audio_directory + '/oga')):
        batch_convert_oga_to_wav(audio_directory)

    # Load and prepare data
    audio_file_paths = load_and_prepare_data(audio_directory + '/wav')

    # Extract metadata
    metadata_df = extract_metadata(audio_file_paths)

    # Check if the results file already exists
    if os.path.exists(output_file_path):
        print(f"Results already exist at {output_file_path}. Loading existing data.")
        features_df = pd.read_csv(output_file_path)
    else:
        # Extract features
        features_df = extract_opensmile_features(audio_file_paths)

    # Merge features with metadata
    results_df = pd.merge(features_df, metadata_df, on='file_path')

    # format final table
    results_df = final_table_format(results_df)

    # Export features to CSV
    export_features_to_csv(results_df, output_file_path)

if __name__ == "__main__":
    # Run the batch conversion
    CORE_DIRECTORY = "/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals"
    AUDIO_DIRECTORY = f"{CORE_DIRECTORY}/data/audio"
    main(CORE_DIRECTORY, AUDIO_DIRECTORY)

