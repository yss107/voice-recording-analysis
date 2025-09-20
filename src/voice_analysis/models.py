"""
Gender Classification Models Module

This module provides machine learning models for predicting gender based on
voice features extracted from audio recordings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os


class GenderClassifier:
    """
    Gender classification system using voice features.
    
    Supports multiple algorithms and provides comprehensive evaluation metrics.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the gender classifier.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False
        
        # Initialize available models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the available machine learning models."""
        
        self.models = {
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.random_state
                ))
            ]),
            
            'gradient_boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=self.random_state
                ))
            ]),
            
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    random_state=self.random_state,
                    probability=True
                ))
            ]),
            
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    C=1.0,
                    solver='liblinear',
                    random_state=self.random_state
                ))
            ]),
            
            'knn': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', KNeighborsClassifier(
                    n_neighbors=5,
                    weights='uniform'
                ))
            ]),
            
            'mlp': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    random_state=self.random_state,
                    max_iter=1000
                ))
            ])
        }
    
    def prepare_features(self, features_list: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features from a list of feature dictionaries.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        if not features_list:
            raise ValueError("Empty features list provided")
        
        # Get feature names from the first sample
        feature_names = [key for key in features_list[0].keys() 
                        if key not in ['file_path', 'label', 'gender']]
        
        # Create feature matrix
        X = np.zeros((len(features_list), len(feature_names)))
        
        for i, features in enumerate(features_list):
            for j, feature_name in enumerate(feature_names):
                X[i, j] = features.get(feature_name, 0.0)
        
        # Handle NaN and infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.feature_names = feature_names
        return X, feature_names
    
    def train_single_model(self, 
                          X: np.ndarray, 
                          y: np.ndarray, 
                          model_name: str) -> Dict[str, Any]:
        """
        Train a single model and return performance metrics.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_name: Name of the model to train
            
        Returns:
            Dictionary with model and performance metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        # Train on full dataset
        model.fit(X, y)
        
        # Get predictions for training data
        y_pred = model.predict(X)
        train_accuracy = accuracy_score(y, y_pred)
        
        return {
            'model': model,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'train_accuracy': train_accuracy,
            'cv_scores': cv_scores
        }
    
    def train_all_models(self, 
                        X: np.ndarray, 
                        y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models and compare their performance.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary with results for all models
        """
        results = {}
        
        print("Training multiple models...")
        
        for model_name in self.models.keys():
            print(f"Training {model_name}...")
            try:
                model_results = self.train_single_model(X, y, model_name)
                results[model_name] = model_results
                
                print(f"{model_name} - CV Accuracy: {model_results['cv_mean']:.4f} (+/- {model_results['cv_std']*2:.4f})")
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        # Find best model based on cross-validation score
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
            self.best_model = results[best_model_name]['model']
            self.is_fitted = True
            
            print(f"\nBest model: {best_model_name} (CV Accuracy: {results[best_model_name]['cv_mean']:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, 
                            X: np.ndarray, 
                            y: np.ndarray, 
                            model_name: str = 'random_forest') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_name: Name of the model to tune
            
        Returns:
            Dictionary with tuning results
        """
        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5, 10]
            },
            'svm': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'classifier__kernel': ['rbf', 'linear']
            },
            'logistic_regression': {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['liblinear', 'lbfgs']
            }
        }
        
        if model_name not in param_grids:
            raise ValueError(f"Hyperparameter tuning not available for {model_name}")
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def evaluate_model(self, 
                      model: Any, 
                      X: np.ndarray, 
                      y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        class_report = classification_report(y, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Args:
            model_name: Name of the model (uses best model if None)
            
        Returns:
            Dictionary with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("No model has been trained yet")
        
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            model = self.models[model_name]
        else:
            model = self.best_model
        
        # Get the classifier from the pipeline
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
        else:
            classifier = model
        
        importance_dict = {}
        
        if hasattr(classifier, 'feature_importances_'):
            # Tree-based models
            importances = classifier.feature_importances_
            for i, importance in enumerate(importances):
                if self.feature_names:
                    importance_dict[self.feature_names[i]] = float(importance)
                else:
                    importance_dict[f'feature_{i}'] = float(importance)
        
        elif hasattr(classifier, 'coef_'):
            # Linear models
            coefficients = np.abs(classifier.coef_[0])
            for i, coef in enumerate(coefficients):
                if self.feature_names:
                    importance_dict[self.feature_names[i]] = float(coef)
                else:
                    importance_dict[f'feature_{i}'] = float(coef)
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a prediction for a single sample.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.is_fitted:
            raise ValueError("No model has been trained yet")
        
        # Prepare features
        X = np.zeros((1, len(self.feature_names)))
        for i, feature_name in enumerate(self.feature_names):
            X[0, i] = features.get(feature_name, 0.0)
        
        # Handle NaN and infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Make prediction
        prediction = self.best_model.predict(X)[0]
        
        result = {'prediction': prediction}
        
        # Add probability if available
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X)[0]
            classes = self.best_model.classes_
            
            result['probabilities'] = {}
            for i, class_name in enumerate(classes):
                result['probabilities'][class_name] = float(probabilities[i])
            
            result['confidence'] = float(max(probabilities))
        
        return result
    
    def save_model(self, file_path: str):
        """
        Save the trained model to disk.
        
        Args:
            file_path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("No model has been trained yet")
        
        model_data = {
            'best_model': self.best_model,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder
        }
        
        joblib.dump(model_data, file_path)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str):
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to the saved model
        """
        if not os.path.exists(file_path):
            raise ValueError(f"Model file {file_path} does not exist")
        
        model_data = joblib.load(file_path)
        
        self.best_model = model_data['best_model']
        self.feature_names = model_data['feature_names']
        self.label_encoder = model_data['label_encoder']
        self.is_fitted = True
        
        print(f"Model loaded from {file_path}")
    
    def create_model_comparison_report(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a comparison report for multiple models.
        
        Args:
            results: Dictionary with model training results
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, model_results in results.items():
            comparison_data.append({
                'Model': model_name,
                'CV Mean Accuracy': model_results['cv_mean'],
                'CV Std': model_results['cv_std'],
                'Train Accuracy': model_results['train_accuracy']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('CV Mean Accuracy', ascending=False)
        
        return df