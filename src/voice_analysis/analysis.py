"""
Data Analysis and Visualization Utilities

This module provides utilities for analyzing voice data and visualizing
feature distributions, model performance, and other insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


class VoiceDataAnalyzer:
    """
    Provides comprehensive analysis and visualization capabilities for voice data.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the analyzer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_features_dataframe(self, 
                                features_list: List[Dict[str, float]], 
                                labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a pandas DataFrame from features list.
        
        Args:
            features_list: List of feature dictionaries
            labels: Optional list of labels
            
        Returns:
            DataFrame with features and labels
        """
        df = pd.DataFrame(features_list)
        
        if labels is not None:
            df['gender'] = labels
        
        return df
    
    def plot_feature_distributions(self, 
                                 df: pd.DataFrame, 
                                 features: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot feature distributions by gender.
        
        Args:
            df: DataFrame with features and gender labels
            features: List of features to plot (plots all if None)
            save_path: Path to save the plot
        """
        if 'gender' not in df.columns:
            raise ValueError("DataFrame must contain 'gender' column")
        
        if features is None:
            features = [col for col in df.columns if col != 'gender' and col != 'file_path']
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                
                # Plot distributions for each gender
                for gender in df['gender'].unique():
                    data = df[df['gender'] == gender][feature].dropna()
                    if len(data) > 0:
                        ax.hist(data, alpha=0.7, label=gender, bins=30)
                
                ax.set_xlabel(feature.replace('_', ' ').title())
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_boxplots(self, 
                            df: pd.DataFrame, 
                            features: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Create box plots for features by gender.
        
        Args:
            df: DataFrame with features and gender labels
            features: List of features to plot (plots all if None)
            save_path: Path to save the plot
        """
        if 'gender' not in df.columns:
            raise ValueError("DataFrame must contain 'gender' column")
        
        if features is None:
            features = [col for col in df.columns if col != 'gender' and col != 'file_path']
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                
                # Create box plot
                df_clean = df[[feature, 'gender']].dropna()
                if len(df_clean) > 0:
                    sns.boxplot(data=df_clean, x='gender', y=feature, ax=ax)
                    ax.set_title(f'{feature.replace("_", " ").title()} by Gender')
                    ax.set_xlabel('Gender')
                    ax.set_ylabel(feature.replace('_', ' ').title())
                    ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, 
                              df: pd.DataFrame, 
                              features: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plot correlation matrix of features.
        
        Args:
            df: DataFrame with features
            features: List of features to include (uses all numeric if None)
            save_path: Path to save the plot
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove non-feature columns
            features = [f for f in features if f not in ['gender', 'file_path']]
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr()
        
        # Create plot
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                   mask=mask, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_pca_analysis(self, 
                         df: pd.DataFrame, 
                         features: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> None:
        """
        Perform and plot PCA analysis.
        
        Args:
            df: DataFrame with features and gender labels
            features: List of features to use (uses all numeric if None)
            save_path: Path to save the plot
        """
        if 'gender' not in df.columns:
            raise ValueError("DataFrame must contain 'gender' column")
        
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f not in ['gender', 'file_path']]
        
        # Prepare data
        X = df[features].fillna(0)
        y = df['gender']
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Explained variance ratio
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        ax1.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Cumulative Explained Variance Ratio')
        ax1.set_title('PCA Explained Variance')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: First two principal components
        colors = {'male': 'blue', 'female': 'red'}
        for gender in y.unique():
            mask = y == gender
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors.get(gender, 'gray'), 
                       label=gender, alpha=0.7)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax2.set_title('PCA: First Two Principal Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature contributions to PC1
        pc1_contributions = pca.components_[0]
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': np.abs(pc1_contributions)
        }).sort_values('importance', ascending=True)
        
        ax3.barh(range(len(feature_importance)), feature_importance['importance'])
        ax3.set_yticks(range(len(feature_importance)))
        ax3.set_yticklabels([f.replace('_', ' ') for f in feature_importance['feature']])
        ax3.set_xlabel('Absolute Contribution')
        ax3.set_title('Feature Contributions to PC1')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature contributions to PC2
        pc2_contributions = pca.components_[1]
        feature_importance_pc2 = pd.DataFrame({
            'feature': features,
            'importance': np.abs(pc2_contributions)
        }).sort_values('importance', ascending=True)
        
        ax4.barh(range(len(feature_importance_pc2)), feature_importance_pc2['importance'])
        ax4.set_yticks(range(len(feature_importance_pc2)))
        ax4.set_yticklabels([f.replace('_', ' ') for f in feature_importance_pc2['feature']])
        ax4.set_xlabel('Absolute Contribution')
        ax4.set_title('Feature Contributions to PC2')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print explained variance for first few components
        print("Explained variance ratio for first 5 components:")
        for i in range(min(5, len(pca.explained_variance_ratio_))):
            print(f"PC{i+1}: {pca.explained_variance_ratio_[i]:.4f}")
    
    def plot_model_performance(self, 
                             results: Dict[str, Dict[str, Any]],
                             save_path: Optional[str] = None) -> None:
        """
        Plot model performance comparison.
        
        Args:
            results: Dictionary with model training results
            save_path: Path to save the plot
        """
        # Prepare data
        models = list(results.keys())
        cv_means = [results[model]['cv_mean'] for model in models]
        cv_stds = [results[model]['cv_std'] for model in models]
        train_accs = [results[model]['train_accuracy'] for model in models]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Cross-validation scores with error bars
        x_pos = np.arange(len(models))
        ax1.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Cross-Validation Accuracy')
        ax1.set_title('Model Performance Comparison (CV)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            ax1.text(i, mean + std + 0.01, f'{mean:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Training vs CV accuracy
        ax2.scatter(cv_means, train_accs, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax2.annotate(model.replace('_', ' ').title(), 
                        (cv_means[i], train_accs[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        # Add diagonal line for reference
        min_acc = min(min(cv_means), min(train_accs))
        max_acc = max(max(cv_means), max(train_accs))
        ax2.plot([min_acc, max_acc], [min_acc, max_acc], 'r--', alpha=0.5)
        
        ax2.set_xlabel('Cross-Validation Accuracy')
        ax2.set_ylabel('Training Accuracy')
        ax2.set_title('Training vs CV Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            labels: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, 
                              importance_dict: Dict[str, float],
                              top_n: int = 15,
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_dict: Dictionary with feature importances
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        # Sort and get top features
        sorted_features = sorted(importance_dict.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, importances, alpha=0.7)
        plt.yticks(y_pos, [f.replace('_', ' ').title() for f in features])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, importance in enumerate(importances):
            plt.text(importance + max(importances) * 0.01, i, 
                    f'{importance:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for features by gender.
        
        Args:
            df: DataFrame with features and gender labels
            
        Returns:
            DataFrame with summary statistics
        """
        if 'gender' not in df.columns:
            raise ValueError("DataFrame must contain 'gender' column")
        
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [f for f in numeric_features if f not in ['gender', 'file_path']]
        
        summary_stats = []
        
        for feature in numeric_features:
            for gender in df['gender'].unique():
                data = df[df['gender'] == gender][feature].dropna()
                
                if len(data) > 0:
                    summary_stats.append({
                        'Feature': feature,
                        'Gender': gender,
                        'Count': len(data),
                        'Mean': data.mean(),
                        'Std': data.std(),
                        'Min': data.min(),
                        'Max': data.max(),
                        'Median': data.median(),
                        'Q25': data.quantile(0.25),
                        'Q75': data.quantile(0.75)
                    })
        
        return pd.DataFrame(summary_stats)
    
    def create_interactive_dashboard(self, df: pd.DataFrame) -> None:
        """
        Create an interactive dashboard with plotly.
        
        Args:
            df: DataFrame with features and gender labels
        """
        if not _PLOTLY_AVAILABLE:
            print("Plotly not available. Install plotly for interactive dashboards.")
            return
            
        if 'gender' not in df.columns:
            raise ValueError("DataFrame must contain 'gender' column")
        
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [f for f in numeric_features if f not in ['gender', 'file_path']]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Distribution', 'Box Plot', 
                          'Scatter Plot', 'Feature Correlation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Histogram for first feature
        if len(numeric_features) > 0:
            feature = numeric_features[0]
            for gender in df['gender'].unique():
                data = df[df['gender'] == gender][feature].dropna()
                fig.add_trace(
                    go.Histogram(x=data, name=f'{gender}', opacity=0.7),
                    row=1, col=1
                )
        
        # Plot 2: Box plot
        if len(numeric_features) > 1:
            feature = numeric_features[1]
            for gender in df['gender'].unique():
                data = df[df['gender'] == gender][feature].dropna()
                fig.add_trace(
                    go.Box(y=data, name=f'{gender}'),
                    row=1, col=2
                )
        
        # Plot 3: Scatter plot
        if len(numeric_features) > 2:
            feature1, feature2 = numeric_features[0], numeric_features[1]
            for gender in df['gender'].unique():
                data = df[df['gender'] == gender]
                fig.add_trace(
                    go.Scatter(x=data[feature1], y=data[feature2], 
                             mode='markers', name=f'{gender}', opacity=0.7),
                    row=2, col=1
                )
        
        # Plot 4: Correlation heatmap
        if len(numeric_features) > 2:
            corr_matrix = df[numeric_features[:10]].corr()  # Limit to first 10 features
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, 
                          x=corr_matrix.columns, 
                          y=corr_matrix.index,
                          colorscale='RdBu', zmid=0),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Voice Analysis Dashboard")
        fig.show()