import pandas as pd

def integrate_sentence_level_data(df_features, df_survey_data):
    """
    Integrate sentence level data with survey data and weighted means data.
    """
    # Merge sentence level data with survey data
    df_merged = pd.merge(df_features, df_survey_data, left_on='UserID', right_on='ID', how='left')

    CORE_COLS = ['UserID',
            'RecordingID',
            'Timestamp',
            'TimestampCorrected',
            'RetreatDate',
            'RelativeDate',
            'PrePost',
            'SentenceID',
            'Sentence',
            'Segment_1']
    
    # add liwc_ prefix to all columns except CORE_COLS and if goemo_ not in col name
    df_merged.columns = ['liwc_' + col if col not in CORE_COLS and 'goemo_' not in col and 'survey_' not in col else col for col in df_merged.columns]

    return df_merged

def integrate_all_means_data(df_all_means, df_survey_data):
    """
    Integrate all means data with sentence level data.
    """
    df_merged = pd.merge(df_all_means, df_survey_data, left_on='UserID', right_on='ID', how='left')

    return df_merged



# SURVEY DATA
df_survey_data = pd.read_csv('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/raw/survey_data.csv')
# Add 'survey_' prefix to all columns except 'ID'
df_survey_data = df_survey_data.rename(columns={col: f'survey_{col}' for col in df_survey_data.columns if col != 'ID'})

# SENTENCE LEVEL DATA
df_features = pd.read_csv('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/journals_sentences_liwc_goemo.csv')

# WEIGHTED MEANS & ACOUSTIC MEANS DATA
df_all_means = pd.read_csv('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/all_mean_scores.csv')


####################################

BASE_DIR = '/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals'

# PROCESSING
df_sentence_level_integrated = integrate_sentence_level_data(df_features, df_survey_data)
# to csv 
df_sentence_level_integrated.to_csv(f'{BASE_DIR}/data/final/sentence_level.csv', index=False)

df_all_means_integrated = integrate_all_means_data(df_all_means, df_survey_data)

# to csv
df_all_means_integrated.to_csv(f'{BASE_DIR}/data/final/means_level.csv', index=False)



