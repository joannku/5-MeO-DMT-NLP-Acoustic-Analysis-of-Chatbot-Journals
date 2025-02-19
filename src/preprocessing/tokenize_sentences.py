import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import pandas as pd

def split_into_sentences(text):
    return sent_tokenize(text)

def generate_sentence_info(row):
    sentences = row['sentences']
    recording_id = row['RecordingID']
    
    # Create tuples of (sentence, sentence_id, position) for each sentence
    sentence_info = [
        {
            'sentence': sent,
            'sentence_id': f"{recording_id}_{str(i+1).zfill(3)}",
            'sentence_position': i+1
        }
        for i, sent in enumerate(sentences)
    ]
    return sentence_info

def tokenize_sentences(input_path, output_path):

    df = pd.read_csv(input_path)

    # Split text into sentences
    df['sentences'] = df['AnonymisedTranscript'].apply(split_into_sentences)

    # Generate sentence info (includes sentence_id and position)
    df['sentence_info'] = df.apply(generate_sentence_info, axis=1)

    # Explode the sentence_info into separate rows
    df_expanded = df.explode('sentence_info')

    # Extract the sentence info dictionary into separate columns
    df_expanded['Sentence'] = df_expanded['sentence_info'].apply(lambda x: x['sentence'])
    df_expanded['SentenceID'] = df_expanded['sentence_info'].apply(lambda x: x['sentence_id'])

    # Clean up intermediate columns
    df_expanded = df_expanded.drop(['sentences', 'sentence_info', 'TranscriptClean', 'AnonymisedTranscript'], axis=1)

    # reorder 
    df_expanded = df_expanded[['UserID', 'RecordingID', 'Timestamp', 'TimestampCorrected', 'RetreatDate', 'RelativeDate', 'PrePost', 'SentenceID', 'Sentence']]

    # Save the processed dataframe
    df_expanded.to_csv(output_path, index=False)

if __name__ == '__main__':
    tokenize_sentences(
        input_path='data/processed/journals_full.csv',
        output_path='data/processed/journals_sentences.csv'
    )

