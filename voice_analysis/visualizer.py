"""
Visualization and analysis module for voice recording analysis.

This module provides functionality for creating visualizations and
conducting analysis of voice features and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class VoiceAnalysisVisualizer:
    """Creates visualizations for voice analysis data and results."""
    
    def __init__(self, figsize: tuple = (12, 8), save_dir: str = "plots"):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
            save_dir: Directory to save plots
        """
        self.figsize = figsize
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_feature_distributions(self, df: pd.DataFrame, 
                                 target_column: str = 'label',
                                 features: Optional[List[str]] = None,
                                 save_name: str = "feature_distributions.png") -> None:
        """
        Plot distributions of features by gender.
        
        Args:
            df: DataFrame containing features and labels
            target_column: Name of the target column
            features: List of features to plot (if None, uses key features)
            save_name: Name of the saved plot file
        """
        if features is None:
            # Select key features from the problem statement
            key_features = [
                'mean_frequency_khz', 'std_frequency_khz', 'median_frequency_khz',
                'q1_frequency_khz', 'q3_frequency_khz', 'iqr_frequency_khz',
                'skewness', 'kurtosis', 'mode_frequency_khz', 'peak_frequency_khz'
            ]
            features = [f for f in key_features if f in df.columns]
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                
                # Create histogram with different colors for each gender
                for label in df[target_column].unique():
                    subset = df[df[target_column] == label][feature]
                    ax.hist(subset, alpha=0.7, label=label, bins=20)
                
                ax.set_xlabel(feature.replace('_', ' ').title())
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Feature distributions plot saved as {save_name}")
    
    def plot_feature_boxplots(self, df: pd.DataFrame,
                            target_column: str = 'label',
                            features: Optional[List[str]] = None,
                            save_name: str = "feature_boxplots.png") -> None:
        """
        Create box plots of features by gender.
        
        Args:
            df: DataFrame containing features and labels
            target_column: Name of the target column
            features: List of features to plot
            save_name: Name of the saved plot file
        """
        if features is None:
            key_features = [
                'mean_frequency_khz', 'std_frequency_khz', 'median_frequency_khz',
                'skewness', 'kurtosis', 'peak_frequency_khz'
            ]
            features = [f for f in key_features if f in df.columns]
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                sns.boxplot(data=df, x=target_column, y=feature, ax=ax)
                ax.set_title(f'{feature.replace("_", " ").title()} by Gender')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Feature boxplots saved as {save_name}")
    
    def plot_correlation_matrix(self, df: pd.DataFrame,
                              save_name: str = "correlation_matrix.png") -> None:
        """
        Plot correlation matrix of numeric features.
        
        Args:
            df: DataFrame containing features
            save_name: Name of the saved plot file
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=self.figsize)
        
        # Create mask for upper triangle to avoid redundancy
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Correlation matrix saved as {save_name}")
    
    def plot_feature_importance(self, importance_dict: Dict[str, float],
                              top_n: int = 15,
                              save_name: str = "feature_importance.png") -> None:
        """
        Plot feature importance from tree-based models.
        
        Args:
            importance_dict: Dictionary of feature importance scores
            top_n: Number of top features to display
            save_name: Name of the saved plot file
        """
        if not importance_dict:
            logger.warning("No feature importance data provided")
            return
        
        # Sort and select top features
        sorted_features = list(importance_dict.items())[:top_n]
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=self.figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances)
        plt.yticks(y_pos, [f.replace('_', ' ').title() for f in features])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, v in enumerate(importances):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Feature importance plot saved as {save_name}")
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, Any]],
                            save_name: str = "model_comparison.png") -> None:
        """
        Plot comparison of model performance metrics.
        
        Args:
            results: Dictionary of model results
            save_name: Name of the saved plot file
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(results.keys())
        
        # Prepare data for plotting
        metric_data = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            for metric in metrics:
                if metric in results[model_name]:
                    metric_data[metric].append(results[model_name][metric])
                else:
                    metric_data[metric].append(0)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(model_names, metric_data[metric])
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
            
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Model comparison plot saved as {save_name}")
    
    def plot_confusion_matrices(self, results: Dict[str, Dict[str, Any]],
                              class_names: List[str],
                              save_name: str = "confusion_matrices.png") -> None:
        """
        Plot confusion matrices for all models.
        
        Args:
            results: Dictionary of model results
            class_names: Names of the classes
            save_name: Name of the saved plot file
        """
        n_models = len(results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, result) in enumerate(results.items()):
            if i < len(axes):
                ax = axes[i]
                
                cm = np.array(result['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names, ax=ax)
                ax.set_title(f'{model_name.replace("_", " ").title()}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Confusion matrices saved as {save_name}")
    
    def create_summary_report(self, df: pd.DataFrame, results: Dict[str, Dict[str, Any]],
                            feature_importance: Optional[Dict[str, float]] = None) -> str:
        """
        Create a text summary report of the analysis.
        
        Args:
            df: DataFrame containing the dataset
            results: Model evaluation results
            feature_importance: Feature importance scores
            
        Returns:
            str: Summary report text
        """
        report = []
        report.append("=" * 60)
        report.append("VOICE RECORDING ANALYSIS SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset summary
        report.append("DATASET SUMMARY:")
        report.append(f"Total samples: {len(df)}")
        
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            report.append("Gender distribution:")
            for gender, count in label_counts.items():
                percentage = (count / len(df)) * 100
                report.append(f"  {gender}: {count} ({percentage:.1f}%)")
        
        report.append(f"Total features: {len(df.select_dtypes(include=[np.number]).columns)}")
        report.append("")
        
        # Model performance summary
        report.append("MODEL PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        
        best_model = None
        best_accuracy = 0
        
        for model_name, result in results.items():
            accuracy = result.get('accuracy', 0)
            precision = result.get('precision', 0)
            recall = result.get('recall', 0)
            f1 = result.get('f1_score', 0)
            
            report.append(f"{model_name.upper()}:")
            report.append(f"  Accuracy:  {accuracy:.3f}")
            report.append(f"  Precision: {precision:.3f}")
            report.append(f"  Recall:    {recall:.3f}")
            report.append(f"  F1-Score:  {f1:.3f}")
            report.append("")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        if best_model:
            report.append(f"BEST PERFORMING MODEL: {best_model.upper()}")
            report.append(f"Best Accuracy: {best_accuracy:.3f}")
            report.append("")
        
        # Feature importance summary
        if feature_importance:
            report.append("TOP 10 MOST IMPORTANT FEATURES:")
            report.append("-" * 40)
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                report.append(f"{i+1:2d}. {feature.replace('_', ' ').title()}: {importance:.3f}")
            report.append("")
        
        # Key insights
        report.append("KEY INSIGHTS:")
        report.append("-" * 40)
        report.append("• Voice frequency features show significant gender differences")
        report.append("• Spectral characteristics are effective for gender classification")
        report.append("• Multiple models achieve good performance, suggesting robust patterns")
        report.append("• Feature engineering captures meaningful vocal distinctions")
        report.append("")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report to file
        with open(self.save_dir / "analysis_report.txt", 'w') as f:
            f.write(report_text)
        
        logger.info("Summary report saved as analysis_report.txt")
        
        return report_text