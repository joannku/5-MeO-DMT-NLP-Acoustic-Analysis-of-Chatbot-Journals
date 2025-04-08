# %%

"""
5-MeO-DMT Journal Language & Voice Analysis Pipeline

This script orchestrates the complete data analysis pipeline for processing and analyzing
5-MeO-DMT journal entries. The pipeline includes data cleaning, feature engineering,
and predictive modeling stages.

Pipeline Stages:
1. Data Preprocessing
2. Feature Engineering
3. Data Integration
4. Predictive Modeling

Author: Joanna Kuc
Institution: UCL
Date: 11/02/2025
"""

from data_cleaning import main as data_cleaning_main
from tokenize_sentences import tokenize_sentences
from weighted_means import main as weighted_means_main
from roberta_go_emotions import main as generate_go_emotions_scores
from vocal_features import main as generate_vocal_features
from integrate_data import main as integrate_data_main

################################################################################
# 1. DATA PREPROCESSING
################################################################################

# %%

def preprocess_data():
    """
    Execute all data preprocessing steps including data cleaning and sentence tokenization.
    """
    # 1.1 Clean and anonymize raw data
    data_cleaning_main()
    """
    Input:
        - data/raw/retreat_dates.csv: Retreat scheduling information
        - data/raw/journals.csv: Raw journal entries
    Output:
        - data/processed/journals_full.csv: Cleaned and anonymized transcripts
    Dependencies:
        - data_cleaning.py
        - anonymise.py
    """
    # TODO: Retreat Dates & Times need fixing - Waiting on George
    
    # 1.2 Tokenize cleaned transcripts into sentences
    tokenize_sentences(
        input_path='/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/journals_full.csv',
        output_path='/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/journals_sentences.csv'
    )
    """
    Input:
        - data/processed/journals_full.csv: Cleaned and anonymized transcripts
    Output:
        - data/processed/journals_sentences.csv: Transcripts tokenized into sentences
    Dependencies:
        - None
    """

################################################################################
# 2. FEATURE ENGINEERING
################################################################################

def engineer_features():
    """
    Generate linguistic, emotional, and acoustic features from processed data.
    
    Features generated:
    - LIWC-22 scores (linguistic features) - requires licensed software
    - GoEmotions scores (emotional features)
    - Acoustic features
    """
    # 1) LIWC
    # NOTE: LIWC processing requires licensed software
    # Process only sentence-level data:
    #   - INPUT: data/processed/journals_sentences.csv
    #   - OUTPUT: data/processed/journals_sentences_liwc.csv
    
    # 2) GoEmotions
    # generate_go_emotions_scores()
    # - INPUT: data/processed/journals_sentences.csv
    # - OUTPUT: data/processed/roberta_go_emotions.csv

    # 3) Vocal Features
    # The original audio files are in oga format, so the function below first converts them to wav format
    # And then extracts features from the wav files using OpenSMILE
    # - INPUT: data/audio/oga -> data/audio/wav
    # - OUTPUT: data/processed/vocal_features.csv
    # generate_vocal_features()

# ################################################################################
# # 3. WEIGHTED MEANS, MEANS AND DATA INTEGRATION
# ################################################################################

weighted_means_main()
    # This function first concats the LIWC scores with the GoEmotions scores
    # And then calculates the weighted means for Pre and Post for the LIWC and GoEmotions
    # - INPUT: data/processed/roberta_go_emotions.csv, data/processed/journals_sentences_liwc.csv
    # - INTERMEDIATE: data/processed/journals_sentences_liwc_goemo.csv
    #       ^ this file can be used for mixed effects models to analyse pre/post effects
    # - OUTPUT: data/processed/sentence_weighted_means.csv

integrate_data_main()

# ################################################################################
# # 4. PREDICTIVE MODELING
# ################################################################################

# def build_prediction_models():
#     """
#     Build and evaluate predictive models for key outcomes.
    
#     Target Variables:
#     1. Psychedelic preparedness
#     2. Subjective experience qualities
#         - Oceanic boundlessness
#         - Emotional breakthrough
#     3. Wellbeing improvement
    
#     Features Used:
#     - PCA emotional profiles
#     - LIWC scores
#     - Acoustic features
#     """
#     # TODO: Implement prediction modeling
#     pass

# %%

if __name__ == "__main__":


    preprocess_data()
    engineer_features()
#     integrate_data()
#     build_prediction_models()




# %%
