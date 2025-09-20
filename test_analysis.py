#!/usr/bin/env python3
"""
Simple test script for voice recording analysis without heavy dependencies.

This script demonstrates the core functionality using sample data.
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to import required modules and provide fallbacks
try:
    import pandas as pd
except ImportError:
    print("pandas not available, using basic data structures")
    pd = None

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    sklearn_available = True
except ImportError:
    print("scikit-learn not available, using basic implementation")
    sklearn_available = False

def create_sample_data():
    """Create sample voice data for testing."""
    print("Creating sample voice data...")
    
    np.random.seed(42)
    n_samples_per_gender = 50
    
    # Male voice characteristics (lower frequencies)
    male_features = {
        'mean_frequency_khz': np.random.normal(0.12, 0.02, n_samples_per_gender),
        'std_frequency_khz': np.random.normal(0.05, 0.01, n_samples_per_gender),
        'median_frequency_khz': np.random.normal(0.11, 0.02, n_samples_per_gender),
        'peak_frequency_khz': np.random.normal(0.20, 0.04, n_samples_per_gender),
        'skewness': np.random.normal(0.5, 0.3, n_samples_per_gender),
        'kurtosis': np.random.normal(2.0, 0.5, n_samples_per_gender),
    }
    
    # Female voice characteristics (higher frequencies)
    female_features = {
        'mean_frequency_khz': np.random.normal(0.18, 0.03, n_samples_per_gender),
        'std_frequency_khz': np.random.normal(0.07, 0.015, n_samples_per_gender),
        'median_frequency_khz': np.random.normal(0.17, 0.03, n_samples_per_gender),
        'peak_frequency_khz': np.random.normal(0.30, 0.06, n_samples_per_gender),
        'skewness': np.random.normal(0.3, 0.4, n_samples_per_gender),
        'kurtosis': np.random.normal(2.2, 0.6, n_samples_per_gender),
    }
    
    # Combine data
    all_features = {}
    for key in male_features.keys():
        all_features[key] = np.concatenate([male_features[key], female_features[key]])
    
    # Add labels
    labels = ['male'] * n_samples_per_gender + ['female'] * n_samples_per_gender
    
    return all_features, labels

def basic_analysis(features, labels):
    """Perform basic analysis without pandas."""
    print("\n=== VOICE RECORDING ANALYSIS ===")
    print(f"Total samples: {len(labels)}")
    
    # Count labels
    male_count = labels.count('male')
    female_count = labels.count('female')
    print(f"Male samples: {male_count}")
    print(f"Female samples: {female_count}")
    
    print(f"Features analyzed: {len(features)}")
    
    # Show feature statistics
    print("\nFeature Statistics:")
    for feature_name, values in features.items():
        male_values = values[:male_count]
        female_values = values[male_count:]
        
        print(f"{feature_name}:")
        print(f"  Male:   mean={np.mean(male_values):.3f}, std={np.std(male_values):.3f}")
        print(f"  Female: mean={np.mean(female_values):.3f}, std={np.std(female_values):.3f}")

def sklearn_analysis(features, labels):
    """Perform machine learning analysis with scikit-learn."""
    print("\n=== MACHINE LEARNING ANALYSIS ===")
    
    # Prepare data
    X = np.column_stack(list(features.values()))
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"{name} Accuracy: {accuracy:.3f}")
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
    # Feature importance for Random Forest
    if 'Random Forest' in results:
        rf_model = models['Random Forest']
        feature_names = list(features.keys())
        importances = rf_model.feature_importances_
        
        print("\nFeature Importance (Random Forest):")
        for name, importance in sorted(zip(feature_names, importances), 
                                      key=lambda x: x[1], reverse=True):
            print(f"  {name}: {importance:.3f}")
    
    return results

def main():
    """Main function."""
    print("Voice Recording Analysis Test")
    print("=" * 40)
    
    # Create sample data
    features, labels = create_sample_data()
    
    # Basic analysis
    basic_analysis(features, labels)
    
    # Machine learning analysis (if available)
    if sklearn_available:
        results = sklearn_analysis(features, labels)
        
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_model]['accuracy']
        
        print(f"\n=== SUMMARY ===")
        print(f"Best Model: {best_model}")
        print(f"Best Accuracy: {best_accuracy:.3f}")
        print(f"Total Features: {len(features)}")
        print(f"Dataset Size: {len(labels)} samples")
        
        print("\nModel successfully demonstrates:")
        print("✓ Feature extraction simulation")
        print("✓ Gender classification")
        print("✓ Model comparison")
        print("✓ Feature importance analysis")
        
    else:
        print("\nNote: Install scikit-learn for full machine learning functionality")
        print("pip install scikit-learn pandas matplotlib seaborn")
    
    print("\n✓ Voice recording analysis test completed successfully!")

if __name__ == "__main__":
    main()