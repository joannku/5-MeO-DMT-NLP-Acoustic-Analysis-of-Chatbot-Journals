"""
Analyze correlations between LIWC features and psychometric questionnaires.

Ridge / Lasso Regressions

"""

import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Tuple
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns


# Global constants
QUESTIONNAIRE_DESCRIPTIONS = {
    'survey_aPPS_total': 'Psychedelic Preparedness Scale (aPPS)',
    'survey_EBI': 'Emotional Breakthrough Inventory (EBI)',
    'survey_ASC_OBN': 'Altered States of Consciousness - Oceanic Boundlessness',
    'survey_ASC_DED': 'Altered States of Consciousness - Dread of Ego Dissolution',
    'survey_bSWEBWBS': 'Short Warwick-Edinburgh Mental Well-being Scale'
}

def load_data(filepath: str) -> pd.DataFrame:
    """Load and return the dataset from CSV file."""
    return pd.read_csv(filepath)


def get_liwc_columns(df: pd.DataFrame) -> List[str]:
    """Return list of LIWC column names containing 'Pre' in their names."""
    return [col for col in df.columns if 'liwc_' in col and 'Pre' in col]


def calculate_correlations(df: pd.DataFrame, 
                         liwc_cols: List[str], 
                         psychometric_cols: List[str]) -> pd.DataFrame:
    """Calculate correlations between LIWC and psychometric measures."""
    analysis_df = df[liwc_cols + psychometric_cols]
    return analysis_df.corr()


def interpret_correlation_effect_size(r: float) -> str:
    """
    Interpret the effect size of a correlation coefficient based on Cohen's guidelines.
    
    Args:
        r: Correlation coefficient
        
    Returns:
        String interpretation of effect size
    """
    r_abs = abs(r)
    if r_abs < 0.1:
        return "negligible"
    elif r_abs < 0.3:
        return "small"
    elif r_abs < 0.5:
        return "medium"
    else:
        return "large"


def collect_strong_correlations(correlation_df: pd.DataFrame,
                              liwc_cols: List[str],
                              psychometric_cols: List[str],
                              r_threshold: float) -> Dict[str, List[Tuple[str, float]]]:
    """Collect correlations stronger than the threshold for each psychometric measure."""
    results = {questionnaire: [] for questionnaire in psychometric_cols}
    
    for liwc_col in liwc_cols:
        for psych_measure in psychometric_cols:
            correlation = correlation_df.loc[liwc_col, psych_measure]
            if abs(correlation) > r_threshold:
                results[psych_measure].append((liwc_col, correlation))
    
    return results


def print_results(results: Dict[str, List[Tuple[str, float]]], r_threshold: float) -> None:
    """Print correlation results organized by questionnaire."""
    print(f"\nStrong correlations (|r| > {r_threshold}) between LIWC features "
          f"and questionnaires:\n")
    
    for questionnaire, correlations in results.items():
        print(f"\n{questionnaire}:")
        if correlations:
            # Sort by absolute correlation strength
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            for liwc_feature, corr in correlations:
                effect_size = interpret_correlation_effect_size(corr)
                print(f"  {liwc_feature}: r = {corr:.3f} ({effect_size} effect)")
        else:
            print("  No strong correlations found")


def print_significant_summary(results_df: pd.DataFrame, 
                            questionnaire_descriptions: Dict[str, str]) -> None:
    """
    Print a summary of significant predictors for each questionnaire after FDR correction.
    
    Args:
        results_df: DataFrame containing all regression results
        questionnaire_descriptions: Dictionary mapping questionnaire codes to descriptions
    """
    print("\n" + "="*80)
    print("SUMMARY OF SIGNIFICANT PREDICTORS AFTER FDR CORRECTION")
    print("="*80)
    
    # Group by significance level
    high_sig = results_df[results_df['P_value_FDR'] < 0.001]
    med_sig = results_df[(results_df['P_value_FDR'] < 0.01) & (results_df['P_value_FDR'] >= 0.001)]
    low_sig = results_df[(results_df['P_value_FDR'] < 0.05) & (results_df['P_value_FDR'] >= 0.01)]
    
    total_sig = len(high_sig) + len(med_sig) + len(low_sig)
    
    print(f"\nTotal significant predictors: {total_sig}")
    print(f"  p < 0.001: {len(high_sig)}")
    print(f"  p < 0.01:  {len(med_sig)}")
    print(f"  p < 0.05:  {len(low_sig)}")
    
    for questionnaire in results_df['Questionnaire'].unique():
        print("\n" + "-"*80)
        print(f"\n{questionnaire_descriptions[questionnaire]}:")
        
        # Get significant results for this questionnaire
        q_results = results_df[
            (results_df['Questionnaire'] == questionnaire) & 
            (results_df['P_value_FDR'] < 0.05)
        ].sort_values('P_value_FDR')
        
        if len(q_results) == 0:
            print("  No significant predictors after FDR correction")
            continue
        
        print("\nSignificant predictors:")
        for _, row in q_results.iterrows():
            sig = '***' if row['P_value_FDR'] < 0.001 else '**' if row['P_value_FDR'] < 0.01 else '*'
            direction = "+" if row['Coefficient'] > 0 else "-"
            effect_size = interpret_standardized_beta(row['Coefficient'])
            print(f"  {direction} {row['Predictor']} {sig}")
            print(f"    β = {row['Coefficient']:.3f} ({effect_size} effect), p_FDR = {row['P_value_FDR']:.3e}")


def interpret_standardized_beta(beta: float) -> str:
    """
    Interpret the effect size of a standardized beta coefficient based on Cohen's guidelines.
    
    Args:
        beta: Standardized beta coefficient
        
    Returns:
        String interpretation of effect size
    """
    beta_abs = abs(beta)
    if beta_abs < 0.1:
        return "negligible"
    elif beta_abs < 0.3:
        return "small"
    elif beta_abs < 0.5:
        return "medium"
    else:
        return "large"


def calculate_cohens_f2(r_squared: float, num_predictors: int) -> float:
    """
    Calculate Cohen's f² effect size for multiple regression.
    
    Args:
        r_squared: R-squared value from the regression model
        num_predictors: Number of predictors in the model
        
    Returns:
        Cohen's f² effect size
    """
    if r_squared == 1.0:
        return float('inf')
    return r_squared / (1 - r_squared)


def interpret_cohens_f2(f2: float) -> str:
    """
    Interpret Cohen's f² effect size based on Cohen's guidelines.
    
    Args:
        f2: Cohen's f² value
        
    Returns:
        String interpretation of effect size
    """
    if f2 < 0.02:
        return "negligible"
    elif f2 < 0.15:
        return "small"
    elif f2 < 0.35:
        return "medium"
    else:
        return "large"


def plot_regression_results(results_df: pd.DataFrame, 
                           questionnaire_descriptions: Dict[str, str],
                           save_path: str = None) -> None:
    """
    Create a visualization of significant LIWC predictors for each questionnaire.
    
    Args:
        results_df: DataFrame containing regression results with FDR-corrected p-values
        questionnaire_descriptions: Dictionary mapping questionnaire codes to descriptions
        save_path: Optional path to save the figure
    """
    # Filter for significant results only
    sig_results = results_df[results_df['P_value_FDR'] < 0.05].copy()
    
    if len(sig_results) == 0:
        print("No significant results to plot after FDR correction.")
        return
    
    # Sort by questionnaire and absolute coefficient size
    sig_results['abs_coef'] = sig_results['Coefficient'].abs()
    sig_results = sig_results.sort_values(['Questionnaire', 'abs_coef'], ascending=[True, False])
    
    # Clean up predictor names for display
    sig_results['Predictor_Clean'] = sig_results['Predictor'].str.replace('liwc_', '').str.replace('_Pre', '')
    
    # Count number of questionnaires with significant results
    questionnaires = sig_results['Questionnaire'].unique()
    n_questionnaires = len(questionnaires)
    
    # Set up the figure
    fig, axes = plt.subplots(n_questionnaires, 1, figsize=(10, 4 * n_questionnaires))
    if n_questionnaires == 1:
        axes = [axes]  # Make it iterable
    
    # Color palette
    palette = {
        'positive': '#2B4F81',  # Dark blue for positive coefficients
        'negative': '#A83C3C'   # Muted red for negative coefficients
    }
    
    # Plot each questionnaire
    for i, questionnaire in enumerate(questionnaires):
        ax = axes[i]
        
        # Get data for this questionnaire
        q_data = sig_results[sig_results['Questionnaire'] == questionnaire].copy()
        
        # Determine color based on coefficient sign
        q_data['color'] = q_data['Coefficient'].apply(
            lambda x: palette['positive'] if x > 0 else palette['negative']
        )
        
        # Sort by coefficient value (not absolute value) for better visualization
        q_data = q_data.sort_values('Coefficient')
        
        # Create horizontal bar plot
        bars = ax.barh(q_data['Predictor_Clean'], q_data['Coefficient'], color=q_data['color'], alpha=0.8)
        
        # Add significance markers
        for j, (_, row) in enumerate(q_data.iterrows()):
            sig = '***' if row['P_value_FDR'] < 0.001 else '**' if row['P_value_FDR'] < 0.01 else '*'
            ax.text(
                0 if row['Coefficient'] < 0 else row['Coefficient'],
                j,
                f" {sig}",
                va='center',
                ha='left' if row['Coefficient'] > 0 else 'right',
                fontweight='bold'
            )
        
        # Add effect size labels
        for j, (_, row) in enumerate(q_data.iterrows()):
            effect_size = interpret_standardized_beta(row['Coefficient'])
            ax.text(
                row['Coefficient'] * 0.5,  # Position in middle of bar
                j,
                f"{row['Coefficient']:.2f}\n({effect_size})",
                va='center',
                ha='center',
                color='white' if abs(row['Coefficient']) > 0.3 else 'black',
                fontweight='bold',
                fontsize=9
            )
        
        # Set title and labels
        ax.set_title(f"{questionnaire_descriptions[questionnaire]}\nSignificant LIWC Predictors (FDR < 0.05)", 
                    fontsize=14, pad=20)
        ax.set_xlabel('Standardized Coefficient (β)', fontsize=12)
        
        # Add zero reference line
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add grid
        # ax.grid(axis='x', alpha=0.3)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add R-squared annotation if available
        if 'R_squared' in q_data.columns and len(q_data['R_squared'].unique()) == 1:
            r2 = q_data['R_squared'].iloc[0]
            f2 = calculate_cohens_f2(r2, len(q_data))
            f2_effect = interpret_cohens_f2(f2)
            ax.annotate(
                f"R² = {r2:.3f}\nCohen's f² = {f2:.3f} ({f2_effect} effect)",
                xy=(0.98, 0.02),
                xycoords='axes fraction',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white', alpha=0.8),
                ha='right',
                va='bottom'
            )
    
    # Add legend
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color=palette['positive']),
         plt.Rectangle((0, 0), 1, 1, color=palette['negative'])],
        ['Positive Association', 'Negative Association'],
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0),
        frameon=False
    )
    
    # Add significance legend
    fig.text(
        0.5, -0.02,
        "* p < 0.05   ** p < 0.01   *** p < 0.001   (FDR-corrected)",
        ha='center',
        fontsize=10
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def run_regression_models(df: pd.DataFrame, 
                        results: Dict[str, List[Tuple[str, float]]]) -> pd.DataFrame:
    """
    Run linear regression models for each questionnaire using its correlated LIWC features.
    Includes FDR correction for multiple comparisons.
    
    Args:
        df: DataFrame containing all variables
        results: Dictionary of correlations by questionnaire
        
    Returns:
        pd.DataFrame: DataFrame containing all regression results
    """
    print("\n" + "="*80)
    print("\nLINEAR REGRESSION MODELS FOR EACH PSYCHOMETRIC MEASURE")
    print("="*80)
    
    # Store all p-values for FDR correction
    all_pvalues = []
    all_coefficients = []
    all_predictors = []
    all_questionnaires = []
    all_r_squared = []
    
    # First pass: collect all p-values
    for questionnaire, correlations in results.items():
        if not correlations:
            continue
            
        predictors = [corr[0] for corr in correlations]
        X = df[predictors].copy()
        y = df[questionnaire].copy()
        
        complete_data = pd.concat([X, y], axis=1).dropna()
        X = complete_data[predictors]
        y = complete_data[questionnaire]
        
        X_with_const = sm.add_constant(X)
        
        try:
            # from formula 
            formula = f"{questionnaire} ~ {' + '.join(predictors)}"
            print(formula)
            model = sm.OLS.from_formula(formula, data=complete_data).fit()
            # Skip the constant term (index 0)
            pvalues = model.pvalues[1:].tolist()
            coefficients = model.params[1:].tolist()
            all_pvalues.extend(pvalues)
            all_coefficients.extend(coefficients)
            all_predictors.extend(predictors)
            all_questionnaires.extend([questionnaire] * len(predictors))
            all_r_squared.extend([model.rsquared] * len(predictors))
        except Exception:
            continue
    
    # Apply FDR correction
    _, pvals_corrected, _, _ = multipletests(all_pvalues, method='fdr_bh')
    
    print("\nMULTIPLE COMPARISON CORRECTION")
    print("-" * 40)
    print(f"Total number of tests corrected for: {len(all_pvalues)}")
    print(f"FDR correction method: Benjamini-Hochberg")
    print("-" * 40 + "\n")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Questionnaire': all_questionnaires,
        'Predictor': all_predictors,
        'Coefficient': all_coefficients,
        'P_value': all_pvalues,
        'P_value_FDR': pvals_corrected,
        'R_squared': all_r_squared
    })
    
    # Second pass: print full results with FDR correction
    for questionnaire, correlations in results.items():
        print("\n" + "="*80)
        print(f"\nANALYZING: {QUESTIONNAIRE_DESCRIPTIONS.get(questionnaire, questionnaire)}")
        print(f"Variable name: {questionnaire}")
        print("="*80 + "\n")
        
        if not correlations:
            print("No predictors found with strong enough correlations\n")
            continue
            
        predictors = [corr[0] for corr in correlations]
        X = df[predictors].copy()
        y = df[questionnaire].copy()
        
        # Print data diagnostics
        print("DATA DIAGNOSTICS")
        print("-" * 40)
        print(f"Total number of participants: {len(df)}")
        print(f"Missing values in outcome measure: {y.isna().sum()}")
        print("\nMissing values in predictors:")
        missing_predictors = X.isna().sum()
        for pred, missing in missing_predictors.items():
            print(f"  {pred}: {missing}")
        
        complete_data = pd.concat([X, y], axis=1).dropna()
        n_complete = len(complete_data)
        if n_complete < len(df):
            print(f"\nWARNING: Analysis based on {n_complete} participants")
            print(f"         {len(df) - n_complete} participants removed due to missing data")
        
        X = complete_data[predictors]
        y = complete_data[questionnaire]
        
        # Multicollinearity check
        X_with_const = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                          for i in range(X_with_const.shape[1])]
        
        print("\nMULTICOLLINEARITY CHECK")
        print("-" * 40)
        print("Variance Inflation Factors:")
        print(vif_data)
        
        try:
            model = sm.OLS(y, X_with_const).fit()
            
            # Calculate effect size for the overall model
            f2 = calculate_cohens_f2(model.rsquared, len(predictors))
            f2_effect = interpret_cohens_f2(f2)
            
            print("\nMODEL RESULTS")
            print("-" * 40)
            print(f"Number of participants in analysis: {n_complete}")
            print(f"Number of predictors: {len(predictors)}")
            print("\nModel fit:")
            print(f"  R-squared: {model.rsquared:.3f}")
            print(f"  Adjusted R-squared: {model.rsquared_adj:.3f}")
            print(f"  Cohen's f²: {f2:.3f} ({f2_effect} effect)")
            print(f"  F-statistic: {model.fvalue:.3f} (p-value: {model.f_pvalue:.3e})")
            
            print("\nCoefficients with FDR-corrected p-values:")
            questionnaire_results = results_df[results_df['Questionnaire'] == questionnaire]
            for _, row in questionnaire_results.iterrows():
                sig = '***' if row['P_value_FDR'] < 0.001 else '**' if row['P_value_FDR'] < 0.01 else '*' if row['P_value_FDR'] < 0.05 else ''
                effect_size = interpret_standardized_beta(row['Coefficient'])
                print(f"  {row['Predictor']}: β = {row['Coefficient']:.3f} ({effect_size} effect) "
                      f"(p = {row['P_value']:.3e}, FDR-corrected p = {row['P_value_FDR']:.3e}) {sig}")
            
            print("\nFull model summary:")
            print(model.summary())
            
        except Exception as e:
            print(f"\nError fitting model: {str(e)}")

    # After all models are fitted and results are printed, add:
    print("\n" + "="*80)
    print_significant_summary(results_df, QUESTIONNAIRE_DESCRIPTIONS)
    
    return results_df


def run_regularized_regression(df: pd.DataFrame,
                             liwc_cols: List[str],
                             questionnaire: str,
                             questionnaire_name: str,
                             reg_type: str = 'elastic',
                             top_n: int = 5) -> None:
    """
    Run Ridge, Lasso, or Elastic Net regression for a questionnaire using all LIWC features.
    """
    # Prepare data
    X = df[liwc_cols].copy()
    y = df[questionnaire].copy()
    
    # Remove rows with missing values
    complete_data = pd.concat([X, y], axis=1).dropna()
    X = complete_data[liwc_cols]
    y = complete_data[questionnaire]
    
    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Setup cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # First fit full model to get all coefficients
    if reg_type.lower() == 'ridge':
        alphas = np.logspace(-3, 3, 100)
        model = RidgeCV(alphas=alphas, cv=cv)
    elif reg_type.lower() == 'lasso':
        alphas = np.logspace(-4, 1, 100)
        model = LassoCV(
            cv=cv,
            random_state=42,
            max_iter=10000,
            selection='random',
            n_jobs=-1
        )
    else:  # elastic net
        alphas = np.logspace(-4, 1, 100)
        # Increase minimum l1_ratio to encourage more sparsity
        l1_ratios = np.linspace(0.5, 0.9, 5)  # Changed from (0.1, 0.9, 9)
        model = ElasticNetCV(
            l1_ratio=l1_ratios,
            alphas=alphas,
            cv=cv,
            random_state=42,
            max_iter=10000,
            selection='random',
            n_jobs=-1
        )
    
    # Fit model and get cross-validated R² score
    model.fit(X_scaled, y)
    full_r2 = model.score(X_scaled, y)
    
    # Calculate effect size for the full model
    full_f2 = calculate_cohens_f2(full_r2, len(liwc_cols))
    full_f2_effect = interpret_cohens_f2(full_f2)
    
    # Get coefficients
    coef_dict = dict(zip(liwc_cols, model.coef_))
    
    if reg_type.lower() == 'ridge':
        # Get top N features
        top_features = sorted(coef_dict.items(), 
                            key=lambda x: abs(x[1]), 
                            reverse=True)[:top_n]
        top_feature_names = [f[0] for f in top_features]
        non_zero_coef = dict(top_features)
        
        # Fit new model with only top features
        X_top = X[top_feature_names]
        X_top_scaled = scaler.fit_transform(X_top)
        model_top = RidgeCV(alphas=alphas, cv=cv)
        model_top.fit(X_top_scaled, y)
        top_r2 = model_top.score(X_top_scaled, y)
        top_f2 = calculate_cohens_f2(top_r2, len(top_feature_names))
        top_f2_effect = interpret_cohens_f2(top_f2)
    elif reg_type.lower() == 'lasso':
        # For Lasso, get features with non-zero coefficients
        non_zero_coef = {k: v for k, v in coef_dict.items() if abs(v) > 1e-3}
        
        # If we have non-zero coefficients, fit a model with just those features
        if non_zero_coef:
            selected_features = list(non_zero_coef.keys())
            X_selected = X[selected_features]
            X_selected_scaled = scaler.fit_transform(X_selected)
            model_selected = LassoCV(cv=cv, random_state=42)
            model_selected.fit(X_selected_scaled, y)
            top_r2 = model_selected.score(X_selected_scaled, y)
            top_f2 = calculate_cohens_f2(top_r2, len(selected_features))
            top_f2_effect = interpret_cohens_f2(top_f2)
        else:
            top_r2 = full_r2
            top_f2 = full_f2
            top_f2_effect = full_f2_effect
    else:  # elastic net
        # Use a higher threshold for Elastic Net to select fewer features
        non_zero_coef = {k: v for k, v in coef_dict.items() if abs(v) > 0.1}  # Increased from 1e-3
        
        # If we have non-zero coefficients, fit a model with just those features
        if non_zero_coef:
            selected_features = list(non_zero_coef.keys())
            X_selected = X[selected_features]
            X_selected_scaled = scaler.fit_transform(X_selected)
            model_selected = ElasticNetCV(
                l1_ratio=np.linspace(0.1, 0.9, 9),
                alphas=alphas,
                cv=cv,
                random_state=42,
                max_iter=10000,
                selection='random',
                n_jobs=-1
            )
            model_selected.fit(X_selected_scaled, y)
            top_r2 = model_selected.score(X_selected_scaled, y)
            top_f2 = calculate_cohens_f2(top_r2, len(selected_features))
            top_f2_effect = interpret_cohens_f2(top_f2)
        else:
            top_r2 = full_r2
            top_f2 = full_f2
            top_f2_effect = full_f2_effect
    
    # Print results
    print(f"\n{reg_type.upper()} REGRESSION RESULTS FOR {questionnaire_name}")
    print("-" * 60)
    print(f"Number of participants: {len(y)}")
    print(f"Total number of predictors: {len(liwc_cols)}")
    if reg_type.lower() == 'ridge':
        print(f"Showing top {top_n} strongest predictors")
    else:
        print(f"Number of selected predictors: {len(non_zero_coef)}")
    print(f"Full model R² score (CV): {full_r2:.3f}")
    print(f"Full model Cohen's f²: {full_f2:.3f} ({full_f2_effect} effect)")
    if len(non_zero_coef) > 0:
        print(f"Selected predictors R² score (CV): {top_r2:.3f}")
        print(f"Selected predictors Cohen's f²: {top_f2:.3f} ({top_f2_effect} effect)")
    print(f"Best alpha: {model.alpha_:.3e}")
    if reg_type.lower() == 'elastic':
        print(f"Best l1_ratio: {model.l1_ratio_:.3f}")
    
    print("\nSelected predictors (standardized coefficients):")
    for feat, coef in sorted(non_zero_coef.items(), key=lambda x: abs(x[1]), reverse=True):
        effect_size = interpret_standardized_beta(coef)
        print(f"  {feat}: {coef:.3f} ({effect_size} effect)")


def run_all_regularized_regressions(df: pd.DataFrame, 
                                  liwc_cols: List[str],
                                  psychometric_measures: List[str],
                                  questionnaire_descriptions: Dict[str, str]) -> None:
    """Run both Ridge and Lasso regressions for all questionnaires."""
    print("\n" + "="*80)
    print("REGULARIZED REGRESSION ANALYSES")
    print("Using all LIWC features as predictors")
    print("="*80)
    
    for questionnaire in psychometric_measures:
        desc = questionnaire_descriptions[questionnaire]
        
        # Ridge regression
        run_regularized_regression(df, liwc_cols, questionnaire, desc, 'ridge')
        
        # Lasso regression
        run_regularized_regression(df, liwc_cols, questionnaire, desc, 'lasso')
        
        # Elastic Net regression
        run_regularized_regression(df, liwc_cols, questionnaire, desc, 'elastic')
        
        print("\n" + "="*80)


def plot_adjusted_regression_relationships(df: pd.DataFrame,
                                         results_df: pd.DataFrame,
                                         questionnaire_descriptions: Dict[str, str],
                                         save_path: str = None) -> None:
    """
    Create horizontal arrangement of scatter plots showing adjusted relationships
    between LIWC predictors and outcomes, controlling for other predictors in the model.
    
    Args:
        df: Original DataFrame with all data
        results_df: DataFrame containing regression results with FDR-corrected p-values
        questionnaire_descriptions: Dictionary mapping questionnaire codes to descriptions
        save_path: Optional path to save the figure
    """
    # Filter for significant results only
    sig_results = results_df[results_df['P_value_FDR'] < 0.05].copy()
    
    if len(sig_results) == 0:
        print("No significant results to plot after FDR correction.")
        return
    
    # Group by questionnaire
    questionnaires = sig_results['Questionnaire'].unique()
    
    # For each questionnaire, find the strongest predictor (by absolute coefficient)
    top_predictors = []
    full_models = {}
    
    for q in questionnaires:
        q_results = sig_results[sig_results['Questionnaire'] == q]
        q_results = q_results.sort_values('abs_coef', ascending=False) if 'abs_coef' in q_results.columns else q_results.sort_values(by='Coefficient', key=abs, ascending=False)
        
        if not q_results.empty:
            # Get all predictors for this questionnaire
            all_predictors = q_results['Predictor'].unique().tolist()
            
            # Get complete data for this model
            X = df[all_predictors].copy()
            y = df[q].copy()
            complete_data = pd.concat([X, y], axis=1).dropna()
            
            # Fit the full model
            X_model = complete_data[all_predictors]
            y_model = complete_data[q]
            X_with_const = sm.add_constant(X_model)
            model = sm.OLS(y_model, X_with_const).fit()
            
            # Store the model
            full_models[q] = (model, all_predictors)
            
            # Get the top predictor
            top_pred = q_results.iloc[0]['Predictor']
            top_coef = q_results.iloc[0]['Coefficient']
            top_p_fdr = q_results.iloc[0]['P_value_FDR']
            
            top_predictors.append((q, top_pred, top_coef, top_p_fdr))
    
    # Set up the figure grid - horizontal arrangement
    n_plots = len(top_predictors)
    if n_plots == 0:
        print("No significant predictors found.")
        return
    
    # Create figure with horizontal layout
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 4))
    if n_plots == 1:
        axes = [axes]  # Make it iterable
    
    # Color palette - similar to the PCA figure
    colors = ['#2B4F81', '#3C7A89', '#A83C3C']
    
    # Dictionary for better LIWC feature names
    liwc_feature_names = {
        'liwc_tone_Pre': 'Positive Tone',
        'liwc_tone_pos_Pre': 'Positive Tone Vocabulary',
        'liwc_posemo_Pre': 'Positive Emotion Words',
        'liwc_negemo_Pre': 'Negative Emotion Words',
        'liwc_anx_Pre': 'Anxiety Words',
        'liwc_anger_Pre': 'Anger Words',
        'liwc_sad_Pre': 'Sadness Words',
        'liwc_social_Pre': 'Social Words',
        'liwc_family_Pre': 'Family Words',
        'liwc_friend_Pre': 'Friend Words',
        'liwc_female_Pre': 'Female References',
        'liwc_male_Pre': 'Male References',
        'liwc_cogproc_Pre': 'Cognitive Process Words',
        'liwc_insight_Pre': 'Insight Words',
        'liwc_cause_Pre': 'Causation Words',
        'liwc_discrep_Pre': 'Discrepancy Words',
        'liwc_tentat_Pre': 'Tentative Words',
        'liwc_certain_Pre': 'Certainty Words',
        'liwc_differ_Pre': 'Differentiation Words',
        'liwc_percept_Pre': 'Perceptual Process Words',
        'liwc_see_Pre': 'Seeing Words',
        'liwc_hear_Pre': 'Hearing Words',
        'liwc_feel_Pre': 'Feeling Words',
        'liwc_bio_Pre': 'Biological Process Words',
        'liwc_body_Pre': 'Body Words',
        'liwc_health_Pre': 'Health Words',
        'liwc_sexual_Pre': 'Sexual Words',
        'liwc_ingest_Pre': 'Ingestion Words',
        'liwc_drives_Pre': 'Drive Words',
        'liwc_affiliation_Pre': 'Affiliation Words',
        'liwc_achieve_Pre': 'Achievement Words',
        'liwc_power_Pre': 'Power Words',
        'liwc_reward_Pre': 'Reward Words',
        'liwc_risk_Pre': 'Risk Words',
        'liwc_focuspast_Pre': 'Past Focus Words',
        'liwc_focuspresent_Pre': 'Present Focus Words',
        'liwc_focusfuture_Pre': 'Future Focus Words',
        'liwc_relativ_Pre': 'Relativity Words',
        'liwc_motion_Pre': 'Motion Words',
        'liwc_space_Pre': 'Space Words',
        'liwc_time_Pre': 'Time Words',
        'liwc_work_Pre': 'Work Words',
        'liwc_leisure_Pre': 'Leisure Words',
        'liwc_home_Pre': 'Home Words',
        'liwc_money_Pre': 'Money Words',
        'liwc_relig_Pre': 'Religion Words',
        'liwc_death_Pre': 'Death Words',
        'liwc_informal_Pre': 'Informal Language',
        'liwc_swear_Pre': 'Swear Words',
        'liwc_netspeak_Pre': 'Netspeak',
        'liwc_assent_Pre': 'Assent Words',
        'liwc_nonflu_Pre': 'Nonfluencies',
        'liwc_filler_Pre': 'Filler Words',
        'liwc_quantity_Pre': 'Quantity Words',
        'liwc_mental_Pre': 'Mental Process Words'
    }
    
    # Dictionary for better questionnaire names and y-axis labels
    questionnaire_labels = {
        'survey_aPPS_total': ('Psychedelic Preparedness Scale', 'PPS Score'),
        'survey_EBI': ('Emotional Breakthrough Inventory', 'EBI Score'),
        'survey_ASC_OBN': ('Oceanic Boundlessness', 'ASC-OBN Score'),
        'survey_ASC_DED': ('Dread of Ego Dissolution', 'ASC-DED Score'),
        'survey_bSWEBWBS': ('Post-Experience Wellbeing', 'sWEMWBS Score')
    }
    
    # Plot each relationship
    for i, (questionnaire, predictor, coef, p_fdr) in enumerate(top_predictors):
        ax = axes[i]
        
        # Get the full model and all predictors
        model, all_predictors = full_models[questionnaire]
        
        # Get data for this relationship
        data_subset = df[all_predictors + [questionnaire]].dropna()
        
        # Calculate adjusted values for the plot
        # 1. Get all predictors except the one we're focusing on
        other_predictors = [p for p in all_predictors if p != predictor]
        
        # 2. Create a design matrix with only the other predictors
        if other_predictors:
            X_others = data_subset[other_predictors]
            X_others_with_const = sm.add_constant(X_others)
            
            # 3. Fit a model predicting the outcome from other predictors
            other_model = sm.OLS(data_subset[questionnaire], X_others_with_const).fit()
            
            # 4. Get residuals (outcome adjusted for other predictors)
            y_adjusted = other_model.resid
            
            # 5. Fit a model predicting the focal predictor from other predictors
            pred_model = sm.OLS(data_subset[predictor], X_others_with_const).fit()
            
            # 6. Get residuals (predictor adjusted for other predictors)
            x_adjusted = pred_model.resid
        else:
            # If there are no other predictors, use the raw values
            x_adjusted = data_subset[predictor]
            y_adjusted = data_subset[questionnaire]
        
        # Create scatter plot of adjusted values
        ax.scatter(x_adjusted, y_adjusted, color=colors[i % len(colors)], alpha=0.6, s=80, edgecolor='white')
        
        # Add regression line for adjusted values
        X_adj_with_const = sm.add_constant(x_adjusted)
        adj_model = sm.OLS(y_adjusted, X_adj_with_const).fit()
        
        # Generate predictions for line
        x_range = np.linspace(x_adjusted.min(), x_adjusted.max(), 100)
        X_pred = sm.add_constant(x_range)
        y_pred = adj_model.predict(X_pred)
        
        # Plot regression line
        ax.plot(x_range, y_pred, color=colors[i % len(colors)], linewidth=2)
        
        # Add confidence interval
        pred = adj_model.get_prediction(X_pred)
        ci = pred.conf_int(alpha=0.05)
        ax.fill_between(x_range, ci[:, 0], ci[:, 1], color=colors[i % len(colors)], alpha=0.2)
        
        # Get better feature name for display
        predictor_clean = liwc_feature_names.get(predictor, predictor.replace('liwc_', '').replace('_Pre', ''))
        
        # Get better questionnaire name and y-axis label
        title, y_label = questionnaire_labels.get(questionnaire, 
                                                (questionnaire_descriptions[questionnaire], 
                                                 questionnaire.replace('survey_', '')))
        
        # Set title and labels
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel(f'{predictor_clean} \n(adjusted)', fontsize=12)
        ax.set_ylabel(f'{y_label} \n(adjusted)', fontsize=12)
        
        # Add statistics annotation
        r_adj = model.rsquared_adj
        p_val = model.pvalues[predictor]
        
        stats_text = (
            f"R²adj = {r_adj:.3f}\n"
            f"β = {coef:.3f}\n"
            f"p = {p_val:.3f}\n"
            f"FDR = {p_fdr:.3f}\n"
            f"BIC = {model.bic:.1f}"
        )
        
        # Position stats box based on coefficient
        if coef > 0:
            xy = (0.75, 0.35)
        else:
            xy = (0.75, 0.95)
            
        ax.annotate(
            stats_text,
            xy=xy,
            xycoords='axes fraction',
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white', alpha=0.8),
            verticalalignment='top'
        )
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


def main():
    # Configuration
    FILEPATH = ('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-'
               'Journals/data/final/means_level.csv')
    R_THRESHOLD = 0.6
    PSYCHOMETRIC_MEASURES = [
        'survey_aPPS_total',
        'survey_EBI',
        'survey_ASC_OBN',
        'survey_ASC_DED',
        'survey_bSWEBWBS'
    ]
    
    # Analysis flags
    RUN_OLS = True
    RUN_RIDGE = False
    RUN_LASSO = False
    RUN_ELASTIC = False

    # Analysis pipeline
    df = load_data(FILEPATH)
    liwc_cols = get_liwc_columns(df)
    correlation_df = calculate_correlations(df, liwc_cols, PSYCHOMETRIC_MEASURES)
    results = collect_strong_correlations(
        correlation_df, 
        liwc_cols, 
        PSYCHOMETRIC_MEASURES, 
        R_THRESHOLD
    )
    print_results(results, R_THRESHOLD)
    
    # Run OLS regression models
    if RUN_OLS:
        results_df = run_regression_models(df, results)
        
        # Add absolute coefficient column for sorting
        results_df['abs_coef'] = results_df['Coefficient'].abs()
        
        # Create and save visualization of regression results
        plot_regression_results(
            results_df, 
            QUESTIONNAIRE_DESCRIPTIONS,
            save_path=('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-'
                      'Journals/outputs/figures/liwc_regression_results.png')
        )
        
        # Create and save scatter plots with adjusted regression lines
        plot_adjusted_regression_relationships(
            df,
            results_df,
            QUESTIONNAIRE_DESCRIPTIONS,
            save_path=('/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-'
                      'Journals/outputs/figures/liwc_adjusted_regression_relationships.png')
        )
    
    # Run regularized regression models
    if RUN_RIDGE or RUN_LASSO or RUN_ELASTIC:
        print("\n" + "="*80)
        print("REGULARIZED REGRESSION ANALYSES")
        print("Using all LIWC features as predictors")
        print("="*80)
        
        for questionnaire in PSYCHOMETRIC_MEASURES:
            desc = QUESTIONNAIRE_DESCRIPTIONS[questionnaire]
            
            if RUN_RIDGE:
                run_regularized_regression(df, liwc_cols, questionnaire, desc, 'ridge')
            
            if RUN_LASSO:
                run_regularized_regression(df, liwc_cols, questionnaire, desc, 'lasso')
                
            if RUN_ELASTIC:
                run_regularized_regression(df, liwc_cols, questionnaire, desc, 'elastic')
            
            if RUN_RIDGE or RUN_LASSO or RUN_ELASTIC:
                print("\n" + "="*80)


if __name__ == "__main__":
    main()

