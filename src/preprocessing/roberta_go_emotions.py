import pandas as pd
import datetime as datetime
from transformers import pipeline
from dotenv import load_dotenv
import os
from tqdm import tqdm

def rate_emotions(model_input, roberta_go_emotions_pipeline):
    """
    Rate the emotions of a given text using the RoBERTa GoEmotions model.

    Args:
        model_input (str): The text to rate.

    Returns:
        list: A list of dictionaries containing the emotion labels and their scores.
    """
    # The pipeline expects the actual text, so we directly pass the input string
    return roberta_go_emotions_pipeline(model_input)

def generate_emotions_df(df, roberta_go_emotions_pipeline):
    """
    Generate a dataframe of emotions for each sentence in the input dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the sentences to rate.
        roberta_go_emotions_pipeline: The pipeline for emotion classification.
    """

    # Initialize an empty list to store the rows
    results_rows = []
    
    # Use tqdm to create a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Sentences", unit="sentence"):
    
        try:
            score_dict = {
                'SentenceID': row['SentenceID']
            }

            # Process with roberta_go_emotions
            roberta_output = rate_emotions(row['Sentence'], roberta_go_emotions_pipeline)
            for emotion in roberta_output[0]:
                score_dict['goemo_' + emotion['label']] = emotion['score']
            
            results_rows.append(score_dict)

        except Exception as e:
            print(f"{datetime.datetime.now()} - Error processing row {index} Error: {e}")

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(results_rows)
    return results_df

def export_emotions_df(results_df, file_path='data/analysis/roberta_goemo_results.csv'):
    """
    Export the emotions dataframe to a CSV file.

    Args:
        results_df (pd.DataFrame): The dataframe containing the emotions data.
        file_path (str): The file path where the CSV will be saved.
    """
    results_df.to_csv(file_path, index=False)
    print(f"Completed export of emotions data to {file_path}")

def main():
    """
    Main function to run the RoBERTa GoEmotions analysis.
    """
    # load dotenv
    env = load_dotenv()
    headers = {"Authorization": os.getenv("HUGGINGFACE_TOKEN")}
    roberta_go_emotions_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    # Process the data
    df = pd.read_csv('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/journals_sentences.csv')
    results_df = generate_emotions_df(df, roberta_go_emotions_pipeline)
    export_emotions_df(results_df, file_path='/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/roberta_go_emotions.csv')

if __name__ == "__main__":
    main()