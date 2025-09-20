"""
Machine learning models module for voice recording analysis.

This module implements various machine learning models for gender prediction
based on vocal features.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceGenderClassifier:
    """Machine learning models for gender classification based on voice features."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the classifier.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_fitted = False
        
    def prepare_data(self, df: pd.DataFrame, 
                    target_column: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning models.
        
        Args:
            df: DataFrame containing features and labels
            target_column: Name of the target column
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
        """
        # Remove non-numeric columns except the target
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_columns:
            numeric_columns.remove(target_column)
        
        # Store feature columns for later use
        self.feature_columns = numeric_columns
        
        # Extract features and labels
        X = df[numeric_columns].values
        y = df[target_column].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        logger.info(f"Prepared data with {X.shape[0]} samples and {X.shape[1]} features")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion of test data
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, 
                              random_state=self.random_state, stratify=y)
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize machine learning models.
        
        Returns:
            Dict[str, Any]: Dictionary of initialized models
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True
            )
        }
        
        return models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Train all models and return cross-validation scores.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dict[str, Dict[str, float]]: Cross-validation scores for each model
        """
        # Initialize models
        self.models = self.initialize_models()
        
        # Encode labels and scale features
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        cv_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train the model
            model.fit(X_train_scaled, y_train_encoded)
            
            # Cross-validation
            cv_score = cross_val_score(model, X_train_scaled, y_train_encoded, cv=5)
            cv_scores[name] = {
                'mean_cv_score': np.mean(cv_score),
                'std_cv_score': np.std(cv_score),
                'cv_scores': cv_score.tolist()
            }
            
            logger.info(f"{name} CV Score: {np.mean(cv_score):.3f} (+/- {np.std(cv_score) * 2:.3f})")
        
        self.is_fitted = True
        return cv_scores
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict[str, Dict[str, Any]]: Evaluation metrics for each model
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before evaluation")
        
        # Encode labels and scale features
        y_test_encoded = self.label_encoder.transform(y_test)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            precision = precision_score(y_test_encoded, y_pred, average='weighted')
            recall = recall_score(y_test_encoded, y_pred, average='weighted')
            f1 = f1_score(y_test_encoded, y_pred, average='weighted')
            
            # ROC AUC (only for binary classification)
            if len(np.unique(y_test_encoded)) == 2:
                roc_auc = roc_auc_score(y_test_encoded, y_pred_proba)
            else:
                roc_auc = None
            
            # Classification report
            class_report = classification_report(
                y_test_encoded, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test_encoded, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist()
            }
            
            logger.info(f"{name} Test Accuracy: {accuracy:.3f}")
        
        return results
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Optional[Dict[str, float]]:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model to get feature importance from
            
        Returns:
            Optional[Dict[str, float]]: Feature importance dictionary
        """
        if not self.is_fitted or model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_dict = {
                feature: importance 
                for feature, importance in zip(self.feature_columns, model.feature_importances_)
            }
            
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return None
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                           model_name: str = 'random_forest') -> Dict[str, Any]:
        """
        Tune hyperparameters for a specific model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model to tune
            
        Returns:
            Dict[str, Any]: Best parameters and score
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return {}
        
        # Prepare data
        y_train_encoded = self.label_encoder.transform(y_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Grid search
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train_encoded)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best score for {model_name}: {grid_search.best_score_:.3f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_models(self, save_dir: str = "models"):
        """
        Save trained models and preprocessing objects.
        
        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, save_path / f"{name}_model.pkl")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, save_path / "scaler.pkl")
        joblib.dump(self.label_encoder, save_path / "label_encoder.pkl")
        
        # Save feature columns
        with open(save_path / "feature_columns.txt", 'w') as f:
            for feature in self.feature_columns:
                f.write(f"{feature}\n")
        
        logger.info(f"Models saved to {save_path}")
    
    def load_models(self, save_dir: str = "models"):
        """
        Load trained models and preprocessing objects.
        
        Args:
            save_dir: Directory to load models from
        """
        save_path = Path(save_dir)
        
        if not save_path.exists():
            raise FileNotFoundError(f"Model directory {save_path} not found")
        
        # Load models
        model_files = list(save_path.glob("*_model.pkl"))
        for model_file in model_files:
            name = model_file.stem.replace("_model", "")
            self.models[name] = joblib.load(model_file)
        
        # Load preprocessing objects
        self.scaler = joblib.load(save_path / "scaler.pkl")
        self.label_encoder = joblib.load(save_path / "label_encoder.pkl")
        
        # Load feature columns
        with open(save_path / "feature_columns.txt", 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]
        
        self.is_fitted = True
        logger.info(f"Models loaded from {save_path}")
    
    def predict(self, X: np.ndarray, model_name: str = 'random_forest') -> np.ndarray:
        """
        Make predictions using a specific model.
        
        Args:
            X: Features to predict
            model_name: Name of the model to use
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted or model_name not in self.models:
            raise ValueError(f"Model {model_name} is not trained")
        
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.models[model_name].predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred