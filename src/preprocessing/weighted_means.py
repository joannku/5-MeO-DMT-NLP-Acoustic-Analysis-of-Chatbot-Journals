# %%

import pandas as pd

def merge_liwc_and_goemo(df_liwc, df_goemo):
    """Merge LIWC and GoEmo features into a single dataframe"""
    return pd.merge(df_liwc, df_goemo, on='SentenceID', how='left')

def calculate_weighted_means(df, features, weight_col='WC'):
    """Calculate weighted means for text features with word count weighting"""
    def safe_weighted_mean(group, feature):
        total_words = group[weight_col].sum()
        if total_words == 0:
            return None
        return (group[feature] * group[weight_col]).sum() / total_words
    
    means = df.groupby('UserID').apply(
        lambda x: pd.Series({
            feature: safe_weighted_mean(x, feature) 
            for feature in features
        })
    )
    return means

def process_weighted_means(df, features, weight_col='WC', lower_bound=-14, mid_bound=0, upper_bound=15):

    pre_data = df[df['RelativeDate'].between(lower_bound, mid_bound)]
    post_data = df[df['RelativeDate'].between(mid_bound, upper_bound)]
    
    # Calculate weighted means for text features
    pre_means = calculate_weighted_means(pre_data, features)
    post_means = calculate_weighted_means(post_data, features)

    # Rename columns to indicate Pre/Post
    pre_means = pre_means.add_suffix('_Pre')
    post_means = post_means.add_suffix('_Post')

    return pd.concat([pre_means, post_means], axis=1)

def get_vocal_means(df_vocal):
    """Calculate means for acoustic features
    One mean for Pre and one for Post"""
    print(df_vocal.columns.tolist())
    COLS_TO_IGNORE = ['UserID', 'RecordingID', 'Timestamp', 'TimestampCorrected', 'RetreatDate', 'RelativeDate', 'PrePost', 'file_path']
    
    # Get potential numeric columns (those not in COLS_TO_IGNORE)
    potential_numeric_cols = [col for col in df_vocal.columns if col not in COLS_TO_IGNORE]
    
    # Check each column for non-numeric values
    for col in potential_numeric_cols:
        # Try to convert to numeric and identify problematic rows
        non_numeric_mask = pd.to_numeric(df_vocal[col], errors='coerce').isna() & ~df_vocal[col].isna()
        if non_numeric_mask.any():
            problematic_rows = df_vocal[non_numeric_mask]
            print(f"\nProblematic values in column '{col}':")
            print(f"Number of problematic rows: {len(problematic_rows)}")
            print("Sample of problematic values:")
            print(problematic_rows[['UserID', col]].head())
    
    # Continue with the rest of the function...
    vocal_features = [col for col in df_vocal.columns if col not in COLS_TO_IGNORE]
    pre_means = df_vocal[df_vocal['PrePost'] == 0].groupby('UserID')[vocal_features].mean()
    post_means = df_vocal[df_vocal['PrePost'] == 1].groupby('UserID')[vocal_features].mean()

    # Rename columns to indicate Pre/Post
    pre_means = pre_means.add_suffix('_Pre')
    post_means = post_means.add_suffix('_Post')

    df_vocal_means = pd.concat([pre_means, post_means], axis=1)
    # reset index
    df_vocal_means = df_vocal_means.reset_index()
    # add vocal_ prefix to apart from UserID
    df_vocal_means.columns = ['vocal_' + col if col != 'UserID' else col for col in df_vocal_means.columns]
    return df_vocal_means

def merge_all_means(df_weighted_means, df_vocal_means):
    """Merge weighted means with vocal means"""
    return pd.merge(df_weighted_means, df_vocal_means, on='UserID', how='left')

def main():

    df_liwc = pd.read_csv('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/journals_sentences_liwc.csv')
    df_goemo = pd.read_csv('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/roberta_go_emotions.csv')

    df = merge_liwc_and_goemo(df_liwc, df_goemo)
    # export
    df.to_csv('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/journals_sentences_liwc_goemo.csv', index=False)
    
    # to deduce all features, exclude 
    TO_EXCLUDE = ['UserID', 'RecordingID', 'Timestamp', 'TimestampCorrected', 'RetreatDate', 'RelativeDate', 'PrePost', 'SentenceID', 'Sentence', 'Segment', 'Segment_1', 'WC']
    GOEMO_FEATURES = [col for col in df.columns if 'goemo_' in col]
    LIWC_FEATURES = [col for col in df.columns if col not in TO_EXCLUDE and col not in GOEMO_FEATURES]
    # update colnames to contain the liwc_prefix
    df.columns = ['liwc_' + col if col in LIWC_FEATURES else col for col in df.columns]
    # add liwc_ prefix to LIWC_FEATURES
    LIWC_FEATURES = ['liwc_' + col for col in LIWC_FEATURES]

    ALL_FEATURES = LIWC_FEATURES + GOEMO_FEATURES

    print(df.columns.tolist())
    weighted_means = process_weighted_means(df, ALL_FEATURES)
    print(weighted_means)

    # reset index
    weighted_means = weighted_means.reset_index()

    # to csv
    weighted_means.to_csv('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/sentence_weighted_means.csv', index=False)

    # Load acoustic data
    df_vocal_features = pd.read_csv('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/vocal_features.csv')
    
    # Calculate acoustic means
    vocal_means_df = get_vocal_means(df_vocal_features)
    vocal_means_df.to_csv('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/vocal_means.csv', index=False)

    # Merge all means
    all_means = merge_all_means(weighted_means, vocal_means_df)
    # to csv
    all_means.to_csv('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals/data/processed/all_mean_scores.csv', index=False)

    return weighted_means

if __name__ == '__main__':
    weighted_means = main()
