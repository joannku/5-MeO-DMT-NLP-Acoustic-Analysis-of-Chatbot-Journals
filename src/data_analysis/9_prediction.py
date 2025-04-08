import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

CORE_DIR = "/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals"
df = pd.read_csv(f"{CORE_DIR}/data/final/means_level.csv")

PSYCHOMETRICS = [
    'survey_aPPS_total',
    'survey_EBI',
    'survey_ASC_OBN',
    'survey_ASC_DED',
    'survey_bSWEBWBS'
]

sig_cats = open(f"{CORE_DIR}/config/mlm_sig_cats.txt", "r").read().splitlines()
print(sig_cats)
# add liwc_ to the beginning of each category
SIG_CATS_PRE = [f"liwc_{cat}_Pre" for cat in sig_cats]
print(SIG_CATS_PRE)

# Method 1: Calculate correlations individually
for cat in SIG_CATS_PRE:
    print(f"\nCorrelations for {cat}:")
    for psychometric in PSYCHOMETRICS:
        corr = df[cat].corr(df[psychometric])
        print(f"{psychometric}: {corr:.4f}")

# Method 2: Create a correlation matrix (more efficient)
print("\n\nCorrelation Matrix:")
corr_matrix = df[SIG_CATS_PRE + PSYCHOMETRICS].corr()
# Extract only the correlations between SIG_CATS_PRE and PSYCHOMETRICS
liwc_psycho_corr = corr_matrix.loc[SIG_CATS_PRE, PSYCHOMETRICS]
print(liwc_psycho_corr)

# Visualize the correlation matrix
plt.figure(figsize=(12, len(SIG_CATS_PRE) * 0.4))
sns.heatmap(liwc_psycho_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlations between LIWC Categories and Psychometric Measures')
plt.tight_layout()
plt.savefig(f"{CORE_DIR}/results/figures/liwc_psychometrics_correlations.png")
plt.show()

# You could also identify the strongest correlations
print("\nTop correlations (absolute value):")
corr_unstack = liwc_psycho_corr.unstack().sort_values(key=abs, ascending=False)
print(corr_unstack.head(10))
