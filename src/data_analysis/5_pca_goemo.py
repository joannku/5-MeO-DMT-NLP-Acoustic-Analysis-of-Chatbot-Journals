# %%

import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests, fdrcorrection
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from IPython.display import display
import statsmodels.api as sm
from matplotlib.colors import ListedColormap
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
import os
import json

class EmotionPCAAnalyzer:
    """Base class for PCA analysis of emotion features."""
    
    def __init__(self, data: pd.DataFrame, goemo_cols: list, liwc_cols: list, survey_cols: list, vocal_cols: list, features: list):
        """Initialize with preprocessed emotion data."""
        if not isinstance(data, pd.DataFrame) or goemo_cols is None or liwc_cols is None or survey_cols is None:
            raise ValueError("Invalid input: data must be DataFrame and goemo_cols, liwc_cols, and survey_cols must be provided")
            
        self.data = data
        self.goemo_cols = goemo_cols
        self.liwc_cols = liwc_cols
        self.vocal_cols = vocal_cols
        self.features = features
        
        self.pca = None
        self.loadings = None
        self.n_components = None

    def run_pca_analysis(self):
        """Run PCA analysis with fixed number of components."""
        # Combine all feature columns
        self.X = self.data[self.features]
        
        # Print initial shape
        print(f"Initial shape of feature matrix: {self.X.shape}")
        
        # Handle missing values
        # First, check how many missing values we have
        missing_counts = self.X.isnull().sum()
        print("\nColumns with missing values:")
        print(missing_counts[missing_counts > 0])
        
        # Drop columns with more than 50% missing values
        threshold = len(self.X) * 0.5
        columns_to_keep = self.X.columns[self.X.isnull().sum() <= threshold]
        self.X = self.X[columns_to_keep]
        
        # For remaining missing values, use mean imputation
        self.X = self.X.fillna(self.X.mean())
        
        print(f"\nShape after handling missing values: {self.X.shape}")
        
        # Update features list to match cleaned X
        self.features = list(self.X.columns)
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Run PCA with selected number of components
        self.pca = PCA(n_components=self.n_components)
        components = self.pca.fit_transform(X_scaled)
        
        # Calculate loadings
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=self.features
        )
        
        # Add PC scores to data
        pc_scores = pd.DataFrame(
            components,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=self.X.index
        )
        self.data = pd.concat([self.data, pc_scores], axis=1)
        
        # Print variance explained
        print("\nVariance explained by each component:")
        for i, var in enumerate(self.pca.explained_variance_ratio_):
            print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
        print(f"Total variance explained: {sum(self.pca.explained_variance_ratio_):.3f} "
              f"({sum(self.pca.explained_variance_ratio_)*100:.1f}%)")
        
        return self.loadings

    @property
    def components(self):
        """Get PCA components if they exist."""
        if self.pca is None:
            raise ValueError("PCA has not been run yet")
        return self.pca.components_

    @property
    def explained_variance_ratio(self):
        return self.pca.explained_variance_ratio_ if self.pca else None

    def get_significant_loadings(self, threshold=0.2):
        """
        Get loadings above threshold, replacing insignificant ones with empty string.
        
        Args:
            threshold (float): Absolute threshold for significance
            
        Returns:
            pd.DataFrame: DataFrame with significant loadings only
        """
        if self.loadings is None:
            raise ValueError("Run PCA analysis first")
            
        # Create mask for significant values
        mask = abs(self.loadings) > threshold
        
        # Create copy of loadings and replace insignificant values with empty string
        significant = self.loadings.copy()
        significant[~mask] = ''
        
        return significant.round(3)

    def print_component_interpretations(self, threshold=0.2):
        """
        Print a detailed interpretation of each principal component.
        
        Args:
            threshold (float): Absolute threshold for significant loadings
        """
        if self.loadings is None:
            raise ValueError("Run PCA analysis first")
        
        print("\nPRINCIPAL COMPONENT INTERPRETATIONS")
        print("=" * 50)
        
        for pc_num in range(self.n_components):
            pc_name = f'PC{pc_num + 1}'
            loadings = self.loadings[pc_name]
            
            # Get significant loadings
            sig_pos = loadings[loadings > threshold].sort_values(ascending=False)
            sig_neg = loadings[loadings < -threshold].sort_values(ascending=True)
            
            # Calculate variance explained
            var_explained = self.pca.explained_variance_ratio_[pc_num]
            
            print(f"\n{pc_name}: {var_explained:.1%} of variance")
            print("-" * 50)
            
            # Print positive loadings
            if not sig_pos.empty:
                print("\nPositive loadings (high scores indicate):")
                for emotion, loading in sig_pos.items():
                    emotion_name = emotion.replace('goemo_', '').replace('_Pre', '')
                    print(f"  • {emotion_name:<15} ({loading:.3f})")
            
            # Print negative loadings
            if not sig_neg.empty:
                print("\nNegative loadings (low scores indicate):")
                for emotion, loading in sig_neg.items():
                    emotion_name = emotion.replace('goemo_', '').replace('_Pre', '')
                    print(f"  • {emotion_name:<15} ({loading:.3f})")
            
            # Print interpretation summary
            print("\nInterpretation:")
            print(f"High {pc_name} scores indicate: ", end="")
            high_emotions = [e.replace('goemo_', '').replace('_Pre', '') for e in sig_pos.index]
            print(", ".join(high_emotions) if high_emotions else "no strong positive indicators")
            
            print(f"Low {pc_name} scores indicate: ", end="")
            low_emotions = [e.replace('goemo_', '').replace('_Pre', '') for e in sig_neg.index]
            print(", ".join(low_emotions) if low_emotions else "no strong negative indicators")
            
            print("\nCumulative variance explained: ", 
                  f"{sum(self.pca.explained_variance_ratio_[:pc_num + 1]):.1%}")
            print("=" * 50)

    def save_component_interpretations(self, output_dir, threshold=0.2):
        """
        Save component interpretations to CSV and TXT files.
        
        Args:
            output_dir (str): Directory to save the output files
            threshold (float): Absolute threshold for significant loadings
        """
        if self.loadings is None:
            raise ValueError("Run PCA analysis first")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize lists to store data for CSV
        csv_data = []
        
        # Initialize text for TXT file
        txt_content = "PRINCIPAL COMPONENT INTERPRETATIONS\n" + "=" * 50 + "\n"
        
        for pc_num in range(self.n_components):
            pc_name = f'PC{pc_num + 1}'
            loadings = self.loadings[pc_name]
            
            # Get significant loadings
            sig_pos = loadings[loadings > threshold].sort_values(ascending=False)
            sig_neg = loadings[loadings < -threshold].sort_values(ascending=True)
            
            # Calculate variance explained
            var_explained = self.pca.explained_variance_ratio_[pc_num]
            
            # Add to CSV data
            for emotion, loading in sig_pos.items():
                emotion_name = emotion.replace('goemo_', '').replace('_Pre', '')
                csv_data.append({
                    'Component': pc_name,
                    'Variance_Explained': var_explained,
                    'Feature': emotion_name,
                    'Loading': loading,
                    'Direction': 'Positive'
                })
            
            for emotion, loading in sig_neg.items():
                emotion_name = emotion.replace('goemo_', '').replace('_Pre', '')
                csv_data.append({
                    'Component': pc_name,
                    'Variance_Explained': var_explained,
                    'Feature': emotion_name,
                    'Loading': loading,
                    'Direction': 'Negative'
                })
            
            # Add to TXT content
            txt_content += f"\n{pc_name}: {var_explained:.1%} of variance\n"
            txt_content += "-" * 50 + "\n"
            
            if not sig_pos.empty:
                txt_content += "\nPositive loadings (high scores indicate):\n"
                for emotion, loading in sig_pos.items():
                    emotion_name = emotion.replace('goemo_', '').replace('_Pre', '')
                    txt_content += f"  • {emotion_name:<15} ({loading:.3f})\n"
            
            if not sig_neg.empty:
                txt_content += "\nNegative loadings (low scores indicate):\n"
                for emotion, loading in sig_neg.items():
                    emotion_name = emotion.replace('goemo_', '').replace('_Pre', '')
                    txt_content += f"  • {emotion_name:<15} ({loading:.3f})\n"
            
            # Add interpretation summary
            txt_content += "\nInterpretation:\n"
            txt_content += f"High {pc_name} scores indicate: "
            high_emotions = [e.replace('goemo_', '').replace('_Pre', '') for e in sig_pos.index]
            txt_content += ", ".join(high_emotions) if high_emotions else "no strong positive indicators"
            txt_content += "\n"
            
            txt_content += f"Low {pc_name} scores indicate: "
            low_emotions = [e.replace('goemo_', '').replace('_Pre', '') for e in sig_neg.index]
            txt_content += ", ".join(low_emotions) if low_emotions else "no strong negative indicators"
            txt_content += "\n"
            
            cumulative_var = sum(self.pca.explained_variance_ratio_[:pc_num + 1])
            txt_content += f"\nCumulative variance explained: {cumulative_var:.1%}\n"
            txt_content += "=" * 50 + "\n"
        
        # Save CSV file
        csv_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, 'pca_component_interpretations.csv')
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved CSV interpretations to: {csv_path}")
        
        # Save TXT file
        txt_path = os.path.join(output_dir, 'pca_component_interpretations.txt')
        with open(txt_path, 'w') as f:
            f.write(txt_content)
        print(f"Saved TXT interpretations to: {txt_path}")
        
        return csv_df

class EmotionPCAStatistics(EmotionPCAAnalyzer):
    """Class for statistical analysis of PCA results."""

    def analyze_data(self, dependent_vars, predictors, pc_interpretations=None, alpha=0.05):
        """
        Complete analysis pipeline combining PCA, correlations, and modeling.
        
        Args:
            dependent_vars (list): Variables to predict
            predictors (list): Predictor variables
            pc_interpretations (dict, optional): PC interpretations
            alpha (float, optional): Significance level
            
        Returns:
            tuple: (PCA results, correlation results, model results)
        """
        # Run PCA
        pca_results = self.run_pca_analysis()
        
        # Analyze correlations
        correlations = self.analyze_pc_correlations(
            self.data, 
            predictors, 
            threshold=alpha
        )
        
        # Run and analyze models
        model_results = self.analyze_models(
            dependent_vars, 
            self.data, 
            predictors, 
            pc_interpretations, 
            alpha
        )
        
        return pca_results, correlations, model_results

    def analyze_models(self, dependent_vars, data, predictors, pc_interpretations=None, alpha=0.05):
        """Run and analyze models for all variables."""
        results = []
        
        for dv in dependent_vars:
            try:
                # Fit full model
                model = smf.ols(f"{dv} ~ {' + '.join(predictors)}", data=data).fit()
                
                # Collect results
                for param, coef, pval in zip(model.params.index[1:], model.params[1:], model.pvalues[1:]):
                    result = {
                        'DV': dv,
                        'Predictor': param,
                        'Coefficient': coef,
                        'P_value': pval
                    }
                    if pc_interpretations and param in pc_interpretations:
                        result['Interpretation'] = pc_interpretations[param]
                    results.append(result)
                    
            except Exception as e:
                print(f"Error in model for {dv}: {str(e)}")
                continue
        
        if not results:
            return None
            
        results_df = pd.DataFrame(results)
        _, fdr_p = fdrcorrection(results_df['P_value'], alpha=alpha)
        results_df['FDR_P'] = fdr_p
        results_df['Significant'] = fdr_p < alpha
        
        return results_df.sort_values('P_value')

class EmotionPCAVisualizer(EmotionPCAAnalyzer):
    """Class for visualizing PCA results."""

    def plot_pca_diagnostics(self, save_path=None, n_components=3):
        """Create diagnostic plots with consistent styling."""
        if self.pca is None:
            # Update the PCA initialization to use fixed number of components
            self.n_components = n_components
            self.run_pca_analysis()
            
        colors = {
            'scree': '#2B4F81',      # Dark blue
            'elbow': '#A83C3C',      # Muted red
            'bars': '#3C7A89',       # Teal
            'reference': '#E15759'    # Reference line color
        }
        
        # Get full PCA for scree and elbow plots
        pca_full = PCA()
        X_scaled = StandardScaler().fit_transform(self.X)
        pca_full.fit(X_scaled)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. Scree Plot
        ax1.plot(range(1, len(pca_full.explained_variance_) + 1), 
                pca_full.explained_variance_, 
                color=colors['scree'], marker='o', linestyle='-', linewidth=2)
        ax1.axhline(y=1, color=colors['reference'], linestyle='--', alpha=0.7)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Scree Plot')
        
        # 2. Elbow Plot
        ax2.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
                pca_full.explained_variance_ratio_, 
                color=colors['elbow'], marker='o', linestyle='-', linewidth=2)
        ax2.axvline(x=self.n_components, color=colors['reference'], 
                    linestyle='--', alpha=0.7, 
                    label=f'Selected components (n={self.n_components})')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.set_title('Elbow Plot')
        ax2.legend()
        
        # 3. Bar Plot
        ax3.bar(range(1, self.n_components + 1), 
                self.explained_variance_ratio, 
                color=colors['bars'], alpha=0.8)
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Explained Variance Ratio')
        ax3.set_title('Explained Variance by Selected Components')
        
        # Style improvements
        for ax in [ax1, ax2, ax3]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches='tight')
        
           # Display the plot
        plt.show()
        
        # Print variance information
        self._print_variance_info()
        
        return fig

    def _print_variance_info(self):
        """Print variance information for components."""
        print(f"\nNumber of components with eigenvalue > 1: {self.n_components}")
        print("\nExplained variance ratio for each component:")
        for i, var in enumerate(self.explained_variance_ratio, 1):
            print(f"PC{i}: {var:.3f} ({var*100:.1f}%)")
        
        print("\nCumulative explained variance ratio:")
        cumulative = np.cumsum(self.explained_variance_ratio)
        for i, cum_var in enumerate(cumulative, 1):
            print(f"PC1-PC{i}: {cum_var:.3f} ({cum_var*100:.1f}%)")

    def plot_loadings_heatmap(self, save_path=None, threshold=0.2, figsize=(12, 10), component_labels=None):
        """Create heatmap of PCA loadings with custom component labels.
        
        Args:
            save_path (str, optional): Path to save the figure
            threshold (float, optional): Threshold for significant loadings
            figsize (tuple, optional): Figure size
            component_labels (dict, optional): Dictionary mapping PC names to descriptive labels
        """
        if self.pca is None:
            self.run_pca_analysis()
        
        # Create loadings with simplified column names
        loadings = self.loadings.copy()
        loadings.index = [col.replace('_Pre', '').replace('goemo_', '') for col in self.features]
        
        # Create a copy for display with potentially renamed columns
        display_loadings = loadings.copy()
        
        # Rename columns if component labels are provided
        if component_labels:
            column_mapping = {}
            for col in display_loadings.columns:
                if col in component_labels:
                    column_mapping[col] = component_labels[col]
            
            if column_mapping:
                display_loadings = display_loadings.rename(columns=column_mapping)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create custom colormap for non-significant values
        gray_cmap = ListedColormap(['#E0E0E0'])  # Light gray
        
        # Plot non-significant values in gray
        mask_nonsig = abs(display_loadings) <= threshold
        ax = sns.heatmap(display_loadings.where(mask_nonsig), 
                         mask=~mask_nonsig,
                         cmap=gray_cmap,
                         center=0,
                         annot=True,
                         fmt='.2f',
                         cbar=False)
        
        # Plot significant values with coolwarm colormap
        mask_sig = abs(display_loadings) > threshold
        sns.heatmap(display_loadings.where(mask_sig),
                    mask=~mask_sig,
                    cmap='coolwarm',
                    center=0,
                    annot=True,
                    fmt='.2f',
                    cbar=True,
                    ax=ax)
        
        # plt.title(f'PCA Loadings (colored if |loading| > {threshold})')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add more space at the bottom for the labels
        plt.subplots_adjust(bottom=0.3)
        
        # Adjust layout to ensure labels are visible
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches='tight')
        
        # Make sure to display the plot
        plt.show()
        
        return loadings  # Return the original loadings

    def calculate_emotion_pc_correlations(self):
        """
        Calculate correlations between emotions and selected PCs.
        
        Returns:
            pd.DataFrame: Correlation matrix between emotions and PCs
        """
        if self.pca is None:
            self.run_pca_analysis()
        
        # Get PC scores
        pc_scores = self.data[[f'PC{i+1}' for i in range(self.n_components)]]
        
        # Calculate correlations between emotions and PCs
        correlations = pd.DataFrame(
            np.corrcoef(self.X.T, pc_scores.T)[:len(self.features), -self.n_components:],
            index=[col.replace('_Pre', '').replace('goemo_', '') for col in self.features],
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        return correlations

class PCARegressor:
    """Class for running regression analyses on PCA components."""
    
    def __init__(self, data, pc_interpretations):
        """
        Initialize PCARegressor.
        
        Args:
            data (pd.DataFrame): DataFrame containing PC scores and dependent variables
            pc_interpretations (dict): Dictionary of PC interpretations
        """
        self.data = data
        self.pc_interpretations = pc_interpretations
        
    def run_full_models(self, dependent_vars, predictors):
        """
        Run multiple regression models with all predictors.
        
        Args:
            dependent_vars (list): List of dependent variables to predict
            predictors (list): List of predictor variables
            
        Returns:
            dict: Dictionary of fitted models
        """
        # Ensure PC columns exist in data
        for pc in predictors:
            if pc not in self.data.columns:
                raise ValueError(f"Column {pc} not found in data")
        
        predictor_str = ' + '.join(predictors)
        self.full_models = {}  # Store as instance attribute
        
        for dv in dependent_vars:
            try:
                formula = f"{dv} ~ {predictor_str}"
                print(f"Fitting model: {formula}")
                print(f"Available columns: {self.data.columns.tolist()}")
                self.full_models[dv] = smf.ols(formula, data=self.data).fit()
            except Exception as e:
                print(f"Error fitting model for {dv}: {str(e)}")
                continue

        for model_name, model in self.full_models.items():
            print(f"\nModel: {model_name}")
            print(model.summary())
            
        return self.full_models
    
    def analyze_full_models(self, models, predictors):
        """Apply FDR correction across all p-values from all models."""
        results_list = []
        all_pvals = []
        temp_results = []
        
        # First pass: collect all p-values and store results
        for model_name, model in models.items():
            pvals = model.pvalues[1:].tolist()
            coefs = model.params[1:].tolist()
            
            all_pvals.extend(pvals)
            
            for pred, coef, p_orig in zip(predictors, coefs, pvals):
                temp_results.append({
                    'Model': model_name,
                    'Component': pred,
                    'Coefficient': coef,
                    'Original_P': p_orig,
                    'High PC State': self.pc_interpretations[pred]['High PC State'],
                    'Low PC State': self.pc_interpretations[pred]['Low PC State']
                })
        
        # Apply FDR correction
        _, p_corrected = fdrcorrection(all_pvals, alpha=0.05)
        
        # Second pass: create final results with corrected p-values
        for result, p_fdr in zip(temp_results, p_corrected):
            result['FDR_P'] = p_fdr
            result['Significant'] = p_fdr < 0.05
            results_list.append(result)
        
        self.results_df = pd.DataFrame(results_list)  # Store as instance attribute
        significant_results = self.results_df[self.results_df['Significant']]
        
        return self.results_df.sort_values(['Model', 'Original_P']), significant_results
    
    def run_single_pc_models(self, significant_results):
        """
        Run single predictor models for significant components.
        
        Args:
            significant_results (pd.DataFrame): DataFrame of significant results
            
        Returns:
            tuple: (Dictionary of fitted models, Results DataFrame)
        """
        # Create models for significant components
        self.single_pc_models = {}  # Store as instance attribute
        for _, row in significant_results.iterrows():
            dv = row['Model']
            pc = row['Component']
            model = smf.ols(f"{dv} ~ {pc}", data=self.data).fit()
            self.single_pc_models[dv] = (model, pc)  # Store in instance attribute
        
        # Collect results
        all_p_values = []
        coefficients = []
        model_names = []
        components = []
        r_squared = []
        
        for model_name, (model, pc) in self.single_pc_models.items():
            all_p_values.extend(model.pvalues[1:])
            coefficients.extend(model.params[1:])
            model_names.extend([model_name])
            components.extend([pc])
            r_squared.extend([model.rsquared])
        
        # Apply FDR correction
        _, p_corrected = fdrcorrection(all_p_values, alpha=0.05, method='indep')
        
        # Create results DataFrame
        self.final_results = pd.DataFrame({  # Store as instance attribute
            'Model': model_names,
            'Component': components,
            'High PC State': [self.pc_interpretations[pc]['High PC State'] for pc in components],
            'Low PC State': [self.pc_interpretations[pc]['Low PC State'] for pc in components],
            'Coefficient': coefficients,
            'R_squared': r_squared,
            'Original_P': all_p_values,
            'FDR_P': p_corrected,
            'Significant': p_corrected < 0.05
        })

        return self.single_pc_models, self.final_results.sort_values('Original_P')

    def plot_pc_relationship(self, data, pc, outcome, model, 
                            color='purple', ax=None,
                            title=None,
                            y_label=None,
                            pc_label=None):
        """
        Plot relationship between PC and psychological measure with enhanced styling
        """
        # Verification prints
        print(f"\nPlotting {pc} vs {outcome}")
        print(f"Model summary for {outcome}:")
        print(model.summary().tables[1])
        
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        
        # Set style
        sns.set_style("white")
        
        # Calculate regression line
        x_vals = np.linspace(data[pc].min(), data[pc].max(), 100)
        X_pred = sm.add_constant(pd.Series(x_vals, name=pc))
        y_vals = model.predict(X_pred)

        # Verify data points
        print(f"Number of data points: {len(data)}")
        print(f"X range: {data[pc].min():.2f} to {data[pc].max():.2f}")
        print(f"Y range: {data[outcome].min():.2f} to {data[outcome].max():.2f}")

        # Scatter plot
        ax.scatter(data[pc], data[outcome], color=color, alpha=0.6, s=100)
        
        # Regression line
        ax.plot(x_vals, y_vals, color=sns.dark_palette(color)[3], linestyle='-', linewidth=2)

        # Confidence intervals
        pred_vals = model.get_prediction(X_pred)
        ci = pred_vals.conf_int()
        ax.fill_between(x_vals, ci[:, 0], ci[:, 1], 
                       color=color, alpha=0.2)

        # Labels and title
        if title is None:
            title = f'{outcome} ~ {pc}'
        ax.set_title(title, fontsize=16, pad=20)
        
        # Use the new PC label for x-axis if provided
        if pc_label:
            ax.set_xlabel(pc_label, fontsize=14)
        else:
            ax.set_xlabel(f'{pc} Score', fontsize=14)
        
        ax.set_ylabel(y_label if y_label else outcome, fontsize=14)

        # Get FDR value for this specific relationship
        fdr_val = self.final_results[
            (self.final_results['Model'] == outcome) & 
            (self.final_results['Component'] == pc)
        ]['FDR_P'].values[0]
        
        # Statistics annotation
        p_val = model.pvalues[pc]
        p_text = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        
        stats_text = (
            f"R²adj = {model.rsquared_adj:.3f}\n"
            f"β = {model.params[pc]:.3f}\n"
            f"{p_text}\n"
            f"FDR = {fdr_val:.3f}\n"
            f"BIC = {model.bic:.1f}"
        )
        
        # Position stats box based on coefficient
        if model.params[pc] > 0:
            xy = (0.75, 0.35)
        else:
            xy = (0.75, 0.95)
            
        ax.annotate(stats_text,
                    xy=xy, 
                    xycoords='axes fraction',
                    fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', 
                             edgecolor='gray',
                             facecolor='white',
                             alpha=0.8),
                    verticalalignment='top')
        
        # Adjust spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        return ax  # Just return the axis, don't show the plot

    def create_relationship_plots(self, data, relationships, save_path=None):
        """
        Create a figure with multiple PC relationship plots
        """
        # Create figure with horizontal layout
        number_of_plots = len(relationships)
        x_figsize = number_of_plots * 5
        y_figsize = 4
        fig, axes = plt.subplots(1, number_of_plots, figsize=(x_figsize, y_figsize))
        plt.subplots_adjust(hspace=0.3, bottom=0.2)

        # Define colors
        colors = sns.color_palette("icefire", n_colors=5)

        # Plot each relationship
        for i, (pc, dv, title, y_label, pc_label) in enumerate(relationships):
            print(f"\nProcessing plot {i+1}: {pc} vs {dv}")
            model = self.single_pc_models[dv][0]
            self.plot_pc_relationship(
                data=data,
                pc=pc,
                outcome=dv,
                model=model,
                color=colors[i],
                ax=axes[i],
                title=title,
                y_label=y_label,
                pc_label=pc_label
            )

        plt.tight_layout()
        
        # Display the figure first
        plt.show()
        
        # Then save if a path is provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def create_results_table(self):
        """
        Create a formatted results table combining full and single PC models.
        """
        # Get all DVs from all_psychometrics instead of single_pc_models
        dvs = [
            'survey_aPPS_total',
            'survey_EBI',
            'survey_ASC_OBN',
            'survey_ASC_DED',
            'survey_bSWEBWBS'
        ]
        
        # Create MultiIndex columns
        columns = []
        for dv in dvs:
            columns.append((dv, 'All components'))
            if dv in self.single_pc_models:  # Only add PC column if there's a significant relationship
                _, pc = self.single_pc_models[dv]
                columns.append((dv, pc))
        
        # Create index for metrics and PCs
        pca_cols = [f'PC{i+1}' for i in range(3)]
        index = ['R²', 'BIC', 'Intercept'] + [f'{pc} (p-val, FDR)' for pc in pca_cols]
        
        results = pd.DataFrame(index=index, columns=pd.MultiIndex.from_tuples(columns))
        
        # Fill in values for full models - now for ALL dvs
        for dv in dvs:
            if dv in self.full_models:  # Check if we have results for this DV
                full_model = self.full_models[dv]
                
                results.loc['R²', (dv, 'All components')] = f"{full_model.rsquared:.3f}"
                results.loc['BIC', (dv, 'All components')] = f"{full_model.bic:.3f}"
                results.loc['Intercept', (dv, 'All components')] = f"{full_model.params['Intercept']:.2f}"
                
                for pc in pca_cols:
                    coef = full_model.params[pc]
                    pval = full_model.pvalues[pc]
                    fdr = self.results_df[
                        (self.results_df['Model'] == dv) & 
                        (self.results_df['Component'] == pc)
                    ]['FDR_P'].values[0]
                    
                    results.loc[f'{pc} (p-val, FDR)', (dv, 'All components')] = \
                        f"{coef:.3f} ({pval:.3f}, {fdr:.3f})"
        
        # Fill in values for single PC models (these will only be the significant ones)
        for dv, (model, pc) in self.single_pc_models.items():
            results.loc['R²', (dv, pc)] = f"{model.rsquared:.3f}"
            results.loc['BIC', (dv, pc)] = f"{model.bic:.3f}"
            results.loc['Intercept', (dv, pc)] = f"{model.params['Intercept']:.2f}"
            
            fdr = self.final_results[
                (self.final_results['Model'] == dv) & 
                (self.final_results['Component'] == pc)
            ]['FDR_P'].values[0]
            
            results.loc[f'{pc} (p-val, FDR)', (dv, pc)] = \
                f"{model.params[pc]:.3f} ({model.pvalues[pc]:.3f}, {fdr:.3f})"
        
        return results

########################################################
#################### PCA analysis ######################
########################################################

CORE_DIR = '/Users/joannakuc/5-MeO-DMT-NLP-Acoustic-Analysis-of-Chatbot-Journals'

PSYCHOMETRICS = ['aPPS_total', 'EBI', 'ASC_OBN', 'ASC_DED', 'bSWEBWBS']
PSYCHOMETRICS = [f'survey_{psych}' for psych in PSYCHOMETRICS]

df = pd.read_csv(f'{CORE_DIR}/data/final/means_level.csv')


# remove all users who have no values for any of the psychometrics
df = df[df[PSYCHOMETRICS].notna().all(axis=1)]
# print shape of df
print(f"DataFrame shape: {df.shape}")

display(df.head())

GOEMO_COLS_PRE = [col for col in df.columns if 'goemo_' in col and 'Pre' in col]
LIWC_COLS_PRE = [col for col in df.columns if 'liwc_' in col and 'Pre' in col]
VOCAL_COLS_PRE = [col for col in df.columns if 'vocal_' in col and 'Pre' in col]
SURVEY_COLS = [col for col in df.columns if 'survey_' in col]



print(" ### PCA analysis ### \n")

print("\nChecking column definitions:")
print(f"GOEMO columns: {len(GOEMO_COLS_PRE)}")
print(f"Sample GOEMO cols: {GOEMO_COLS_PRE[:5]}")
print(f"\nLIWC columns: {len(LIWC_COLS_PRE)}")
print(f"Sample LIWC cols: {LIWC_COLS_PRE[:5]}")
print(f"\nVOCAL columns: {len(VOCAL_COLS_PRE)}")
print(f"Sample VOCAL cols: {VOCAL_COLS_PRE[:5]}")

print("\nChecking DataFrame:")
print(f"DataFrame shape: {df.shape}")
print("\nSample columns in DataFrame:")
print(df.columns[:10].tolist())

# Initialize analyzer with all feature columns and set n_components
analyzer = EmotionPCAAnalyzer(
    data=df,
    goemo_cols=GOEMO_COLS_PRE,
    liwc_cols=LIWC_COLS_PRE,
    survey_cols=SURVEY_COLS,
    vocal_cols=VOCAL_COLS_PRE,
    features=GOEMO_COLS_PRE
)
analyzer.n_components = 3  # Set number of components
analyzer.run_pca_analysis()  # Run PCA analysis first

# Create visualizer and run PCA with 3 components
visualizer = EmotionPCAVisualizer(
    data=df,
    goemo_cols=GOEMO_COLS_PRE,
    liwc_cols=LIWC_COLS_PRE,
    survey_cols=SURVEY_COLS,
    vocal_cols=VOCAL_COLS_PRE,
    features=GOEMO_COLS_PRE
)

# Run PCA analysis and create diagnostic plots with 3 components
loadings = visualizer.plot_pca_diagnostics(
    save_path=f'{CORE_DIR}/outputs/figures/pca_diagnostics.png',
    n_components=3
)

# Now save component interpretations
print("\nSaving component interpretations...")
analyzer.save_component_interpretations(
    output_dir=f'{CORE_DIR}/outputs/pca',
    threshold=0.2
)

analyzer.data.to_csv(f'{CORE_DIR}/data/final/means_level+pca.csv', index=False)

pca_data = analyzer.data

########################################################
####### Association Between PCs and Psychometrics ######
########################################################

print(" ### Prediction of Questionnaires by PCs ### \n")

PC_INTERPRETATIONS = json.load(open(f'{CORE_DIR}/config/pca_interpretations.json'))['PC_INTERPRETATIONS']

# Define component labels
component_labels = {
    'PC1': 'PC1: Positive Emotion Engagement',
    'PC2': 'PC2: Negative Emotional States',
    'PC3': 'PC3: Reflective Integration'
}

# Get PC scores from the analyzer
pc_scores = analyzer.data[[f'PC{i+1}' for i in range(3)]]

# Initialize regressor
regressor = PCARegressor(data=pca_data, pc_interpretations=PC_INTERPRETATIONS)

# Define dependent variables (questionnaire items)
pca_cols = [f'PC{i+1}' for i in range(3)]

# Define relationships for visualization (significant ones)
relationships = [
    ('PC1', 'survey_aPPS_total', 'Psychedelic Preparedness', 'PPS Score', component_labels['PC1']),
    ('PC1', 'survey_bSWEBWBS', 'Post-Experience Psychological Wellbeing', 'sWEMWBS Score', component_labels['PC1']),
    ('PC3', 'survey_EBI', 'Emotional Breakthrough', 'EBI Score', component_labels['PC3']),
]

# Define all relationships for the table
all_psychometrics = [
    'survey_aPPS_total',
    'survey_EBI',
    'survey_ASC_OBN',
    'survey_ASC_DED',
    'survey_bSWEBWBS'
]

# Run models for all relationships first
full_models = regressor.run_full_models(all_psychometrics, pca_cols)
results_df, significant_results = regressor.analyze_full_models(full_models, pca_cols)
single_pc_models, final_results = regressor.run_single_pc_models(significant_results)

# Now create and save plots for significant relationships
regressor.create_relationship_plots(
    data=pca_data,
    relationships=relationships,
    save_path=f'{CORE_DIR}/outputs/figures/pc_relationships.png'
)

# Create results table with ALL relationships
results_table = regressor.create_results_table()
print("\nResults Table (All Relationships):")
display(results_table)
results_table.to_csv(f'{CORE_DIR}/outputs/tables/pca_results_table.csv')

# Create and display the loadings heatmap
print("\nGenerating PCA loadings heatmap...")

def plot_heatmap_with_multiline_labels(visualizer, save_path=None, threshold=0.2, figsize=(12, 6), component_labels=None, transpose=True):
    """Create heatmap of PCA loadings with multi-line component labels, optionally transposed."""
    if visualizer.pca is None:
        visualizer.run_pca_analysis()
    
    # Define emotion order based on the dendrogram clustering
    emotion_order = [
        # Positive sentiment cluster
        'amusement',
        'excitement',
        'joy',
        'love',
        'desire',
        'optimism',
        'caring',
        'pride',
        'admiration',
        'gratitude',
        'relief',
        'approval',
        'realization',
        # Ambiguous sentiment cluster
        'surprise',
        'curiosity',
        'confusion',
        # Negative sentiment cluster
        'fear',
        'nervousness',
        'remorse',
        'embarrassment',
        'disappointment',
        'sadness',
        'grief',
        'disgust',
        'anger',
        'annoyance',
        'disapproval'
    ]
    
    # Create loadings with simplified column names
    loadings = visualizer.loadings.copy()
    loadings.index = [col.replace('_Pre', '').replace('goemo_', '') for col in visualizer.features]
    
    # Reorder columns according to emotion_order
    if transpose:
        loadings = loadings.T
        loadings = loadings[emotion_order]
    else:
        loadings = loadings.reindex(emotion_order)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create custom colormaps
    gray_cmap = ListedColormap(['#f0f0f0'])  # Lighter gray for non-significant values
    
    # Use RdBu_r (reversed Red-Blue) colormap which is:
    # - More colorblind friendly
    # - Better for scientific visualization
    # - Red for positive loadings, Blue for negative loadings
    # - Higher contrast than coolwarm
    
    # Plot non-significant values in light gray
    mask_nonsig = abs(loadings) <= threshold
    ax = sns.heatmap(loadings.where(mask_nonsig), 
                     mask=~mask_nonsig,
                     cmap=gray_cmap,
                     center=0,
                     annot=True,
                     fmt='.2f',
                     cbar=False)
    
    # Plot significant values with RdBu_r colormap
    mask_sig = abs(loadings) > threshold
    sns.heatmap(loadings.where(mask_sig),
                mask=~mask_sig,
                cmap='RdBu_r',  # Changed from 'coolwarm' to 'RdBu_r'
                center=0,
                annot=True,
                fmt='.2f',
                cbar=True,
                ax=ax,
                vmin=-0.5,  # Set fixed range for better comparison
                vmax=0.5)   # Set fixed range for better comparison

    # plt.title(f'PCA Loadings (colored if |loading| > {threshold})')
    
    # If transposed and component labels provided, rename the y-axis labels
    if transpose and component_labels:
        # Get current y-tick positions
        tick_locs = ax.get_yticks()
        
        # Set new labels at the tick positions
        ax.set_yticks(tick_locs)
        
        # Create new labels list for y-axis
        new_labels = []
        for i, idx in enumerate(loadings.index):
            if idx in component_labels:
                new_labels.append(component_labels[idx])
            else:
                new_labels.append(idx)
        
        # Set the new y-axis labels with proper alignment
        ax.set_yticklabels(new_labels, rotation=0, ha='right', va='center', fontsize=14)
        
        # Adjust margins for y-labels
        plt.subplots_adjust(left=0.35)  # More space on the left for the labels
    
    # Rotate x-axis labels for better readability if we have a lot of emotions (when transposed)
    if transpose:
        plt.xticks(rotation=75, ha='center', fontsize=14)
    else:
        # Apply custom labels with newlines if provided (for non-transposed version)
        if component_labels:
            # Get current tick positions
            tick_locs = ax.get_xticks()
            
            # Set new labels at the tick positions
            ax.set_xticks(tick_locs)
            
            # Create new labels list 
            new_labels = []
            for i, col in enumerate(loadings.columns):
                if col in component_labels:
                    new_labels.append(component_labels[col])
                else:
                    new_labels.append(col)
            
            # Set the new labels with proper horizontal alignment and rotation
            ax.set_xticklabels(new_labels, rotation=45, ha='center', va='top', fontsize=10)
            
            # Add more space at the bottom for the labels
            plt.subplots_adjust(bottom=0.25)
    
    # Tight layout should be after subplots_adjust
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
    
    # Make sure to display the plot
    plt.show()
    
    return loadings

# Call the custom function with transpose=True
loadings_df = plot_heatmap_with_multiline_labels(
    visualizer,
    save_path=f'{CORE_DIR}/outputs/figures/pca_loadings_heatmap_transposed.png',
    threshold=0.2,
    figsize=(17, 3),  # Wider figure for transposed view
    component_labels=component_labels,
    transpose=True
)

# Optional: Force matplotlib to show all open figures
plt.ion()  # Turn on interactive mode
plt.pause(0.1)  # Small pause to ensure display refreshes

# %%


