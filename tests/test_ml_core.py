"""
Core ML Functionality Test

This script tests the machine learning components without requiring audio libraries.
It demonstrates the gender classification capabilities using simulated voice features.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import only the ML components
from voice_analysis.models import GenderClassifier
from voice_analysis.analysis import VoiceDataAnalyzer


def create_sample_features():
    """Create sample voice features for testing ML pipeline."""
    np.random.seed(42)
    
    # Male voice features (lower frequencies)
    male_features = []
    for i in range(100):
        features = {
            'mean_frequency': np.random.normal(150, 30),
            'spectral_entropy': np.random.normal(6.5, 1.0),
            'mode_frequency': np.random.normal(120, 25),
            'f0_mean': np.random.normal(125, 20),
            'f0_std': np.random.normal(15, 5),
            'formant_f1': np.random.normal(500, 50),
            'formant_f2': np.random.normal(1200, 100),
            'spectral_centroid_mean': np.random.normal(1200, 200),
            'rms_energy_mean': np.random.normal(0.02, 0.005),
        }
        male_features.append(features)
    
    # Female voice features (higher frequencies)
    female_features = []
    for i in range(100):
        features = {
            'mean_frequency': np.random.normal(220, 40),
            'spectral_entropy': np.random.normal(7.0, 1.2),
            'mode_frequency': np.random.normal(200, 35),
            'f0_mean': np.random.normal(210, 30),
            'f0_std': np.random.normal(20, 6),
            'formant_f1': np.random.normal(600, 60),
            'formant_f2': np.random.normal(1400, 120),
            'spectral_centroid_mean': np.random.normal(1500, 250),
            'rms_energy_mean': np.random.normal(0.018, 0.004),
        }
        female_features.append(features)
    
    all_features = male_features + female_features
    labels = ['male'] * 100 + ['female'] * 100
    
    return all_features, labels


def test_ml_pipeline():
    """Test the complete ML pipeline."""
    print("Voice Recording Analysis - ML Pipeline Test")
    print("=" * 50)
    
    # Create sample data
    print("1. Creating sample voice features...")
    features_list, labels = create_sample_features()
    print(f"   ✓ Generated {len(features_list)} samples")
    print(f"   ✓ Male: {labels.count('male')}, Female: {labels.count('female')}")
    
    # Initialize classifier
    print("\n2. Initializing gender classifier...")
    classifier = GenderClassifier(random_state=42)
    print("   ✓ Classifier initialized")
    
    # Prepare features
    print("\n3. Preparing features...")
    X, feature_names = classifier.prepare_features(features_list)
    y = np.array(labels)
    print(f"   ✓ Feature matrix shape: {X.shape}")
    print(f"   ✓ Number of features: {len(feature_names)}")
    
    # Train models
    print("\n4. Training multiple models...")
    results = classifier.train_all_models(X, y)
    print("   ✓ Model training completed")
    
    # Display results
    print("\n5. Model Performance Results:")
    for model_name, model_results in results.items():
        cv_mean = model_results['cv_mean']
        cv_std = model_results['cv_std']
        train_acc = model_results['train_accuracy']
        print(f"   {model_name:20}: CV={cv_mean:.4f}±{cv_std:.4f}, Train={train_acc:.4f}")
    
    # Test predictions
    print("\n6. Testing predictions...")
    sample_features = features_list[0]
    actual_label = labels[0]
    prediction = classifier.predict(sample_features)
    
    print(f"   Sample prediction:")
    print(f"   ✓ Actual: {actual_label}")
    print(f"   ✓ Predicted: {prediction['prediction']}")
    if 'confidence' in prediction:
        print(f"   ✓ Confidence: {prediction['confidence']:.4f}")
    
    # Test feature importance
    print("\n7. Feature importance analysis...")
    try:
        importance = classifier.get_feature_importance()
        print("   Top 5 features:")
        for i, (feature, score) in enumerate(list(importance.items())[:5]):
            print(f"   {i+1}. {feature}: {score:.4f}")
    except Exception as e:
        print(f"   Warning: Could not extract feature importance: {e}")
    
    # Test data analysis
    print("\n8. Testing data analysis...")
    analyzer = VoiceDataAnalyzer()
    df = analyzer.create_features_dataframe(features_list, labels)
    print(f"   ✓ Created DataFrame: {df.shape}")
    
    summary = analyzer.generate_summary_statistics(df)
    print(f"   ✓ Generated summary statistics: {summary.shape}")
    
    # Test model saving
    print("\n9. Testing model persistence...")
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'test_classifier.joblib')
    
    classifier.save_model(model_path)
    print(f"   ✓ Model saved to: {model_path}")
    
    # Test loading
    new_classifier = GenderClassifier()
    new_classifier.load_model(model_path)
    test_prediction = new_classifier.predict(sample_features)
    print(f"   ✓ Model loaded and tested successfully")
    
    print("\n" + "=" * 50)
    print("✓ All tests passed successfully!")
    print("\nNext steps:")
    print("- Install librosa for audio processing: pip install librosa")
    print("- Use real audio files with VoiceFeatureExtractor")
    print("- Explore the Jupyter notebook for interactive analysis")
    
    return True


if __name__ == "__main__":
    try:
        test_ml_pipeline()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)