"""
Complete Voice Analysis Workflow Example

This script demonstrates the complete workflow for voice recording analysis
and gender classification using the voice_analysis package.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from voice_analysis.feature_extraction import VoiceFeatureExtractor
from voice_analysis.preprocessing import AudioPreprocessor
from voice_analysis.models import GenderClassifier
from voice_analysis.analysis import VoiceDataAnalyzer


def create_sample_data():
    """
    Create sample voice features data for demonstration.
    In a real scenario, this would be replaced with actual audio file processing.
    """
    np.random.seed(42)
    
    # Simulate features for male voices (typically lower frequencies)
    n_male = 100
    male_features = []
    
    for i in range(n_male):
        features = {
            'mean_frequency': np.random.normal(150, 30),  # Lower mean frequency
            'spectral_entropy': np.random.normal(6.5, 1.0),
            'mode_frequency': np.random.normal(120, 25),
            'f0_mean': np.random.normal(125, 20),
            'f0_std': np.random.normal(15, 5),
            'f0_range': np.random.normal(80, 20),
            'voiced_fraction': np.random.normal(0.65, 0.1),
            'formant_f1': np.random.normal(500, 50),
            'formant_f2': np.random.normal(1200, 100),
            'formant_f3': np.random.normal(2400, 150),
            'spectral_centroid_mean': np.random.normal(1200, 200),
            'spectral_rolloff_mean': np.random.normal(3000, 400),
            'spectral_bandwidth_mean': np.random.normal(800, 100),
            'zero_crossing_rate_mean': np.random.normal(0.05, 0.01),
            'rms_energy_mean': np.random.normal(0.02, 0.005),
            'duration': np.random.normal(3.0, 1.0),
            'tempo': np.random.normal(120, 20)
        }
        male_features.append(features)
    
    # Simulate features for female voices (typically higher frequencies)
    n_female = 100
    female_features = []
    
    for i in range(n_female):
        features = {
            'mean_frequency': np.random.normal(220, 40),  # Higher mean frequency
            'spectral_entropy': np.random.normal(7.0, 1.2),
            'mode_frequency': np.random.normal(200, 35),
            'f0_mean': np.random.normal(210, 30),
            'f0_std': np.random.normal(20, 6),
            'f0_range': np.random.normal(100, 25),
            'voiced_fraction': np.random.normal(0.70, 0.1),
            'formant_f1': np.random.normal(600, 60),
            'formant_f2': np.random.normal(1400, 120),
            'formant_f3': np.random.normal(2800, 180),
            'spectral_centroid_mean': np.random.normal(1500, 250),
            'spectral_rolloff_mean': np.random.normal(3500, 500),
            'spectral_bandwidth_mean': np.random.normal(900, 120),
            'zero_crossing_rate_mean': np.random.normal(0.07, 0.015),
            'rms_energy_mean': np.random.normal(0.018, 0.004),
            'duration': np.random.normal(3.2, 1.2),
            'tempo': np.random.normal(125, 22)
        }
        female_features.append(features)
    
    # Combine features and labels
    all_features = male_features + female_features
    labels = ['male'] * n_male + ['female'] * n_female
    
    return all_features, labels


def demonstrate_feature_extraction():
    """
    Demonstrate audio feature extraction.
    Note: This requires actual audio files to work.
    """
    print("=== Feature Extraction Demo ===")
    
    # Initialize feature extractor
    extractor = VoiceFeatureExtractor(sample_rate=22050)
    
    print("Feature extractor initialized with:")
    print(f"- Sample rate: {extractor.sample_rate} Hz")
    print(f"- Frame length: {extractor.frame_length}")
    print(f"- Hop length: {extractor.hop_length}")
    
    # Note: In a real scenario, you would process actual audio files like this:
    # features = extractor.extract_all_features('path/to/audio/file.wav')
    # print(f"Extracted features: {features}")
    
    print("Note: To extract features from real audio files, use:")
    print("features = extractor.extract_all_features('path/to/audio/file.wav')")
    print()


def demonstrate_preprocessing():
    """
    Demonstrate audio preprocessing capabilities.
    """
    print("=== Audio Preprocessing Demo ===")
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(sample_rate=22050, target_length=3.0)
    
    print("Audio preprocessor initialized with:")
    print(f"- Sample rate: {preprocessor.sample_rate} Hz")
    print(f"- Target length: {preprocessor.target_length} seconds")
    
    # Note: In a real scenario, you would preprocess audio files like this:
    # audio = preprocessor.load_and_preprocess('path/to/audio/file.wav')
    # print(f"Preprocessed audio shape: {audio.shape}")
    
    print("Note: To preprocess real audio files, use:")
    print("audio = preprocessor.load_and_preprocess('path/to/audio/file.wav')")
    print()


def demonstrate_machine_learning():
    """
    Demonstrate machine learning pipeline with sample data.
    """
    print("=== Machine Learning Demo ===")
    
    # Create sample data
    print("Creating sample voice features data...")
    features_list, labels = create_sample_data()
    
    # Initialize classifier
    classifier = GenderClassifier(random_state=42)
    
    # Prepare features
    print("Preparing features for training...")
    X, feature_names = classifier.prepare_features(features_list)
    y = np.array(labels)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Label distribution: {np.unique(y, return_counts=True)}")
    
    # Train all models
    print("\nTraining multiple models...")
    results = classifier.train_all_models(X, y)
    
    # Display results
    print("\n=== Model Performance Results ===")
    for model_name, model_results in results.items():
        print(f"{model_name}:")
        print(f"  CV Accuracy: {model_results['cv_mean']:.4f} (+/- {model_results['cv_std']*2:.4f})")
        print(f"  Train Accuracy: {model_results['train_accuracy']:.4f}")
    
    # Get feature importance
    print("\n=== Feature Importance ===")
    try:
        importance = classifier.get_feature_importance()
        print("Top 10 most important features:")
        for i, (feature, score) in enumerate(list(importance.items())[:10]):
            print(f"{i+1:2d}. {feature.replace('_', ' ').title()}: {score:.4f}")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
    
    # Test prediction
    print("\n=== Prediction Demo ===")
    sample_features = features_list[0]  # Use first sample
    prediction_result = classifier.predict(sample_features)
    
    print(f"Sample prediction:")
    print(f"  Predicted gender: {prediction_result['prediction']}")
    if 'confidence' in prediction_result:
        print(f"  Confidence: {prediction_result['confidence']:.4f}")
    if 'probabilities' in prediction_result:
        print(f"  Probabilities: {prediction_result['probabilities']}")
    
    return classifier, X, y, labels, features_list


def demonstrate_data_analysis(classifier, X, y, labels, features_list):
    """
    Demonstrate data analysis and visualization.
    """
    print("\n=== Data Analysis Demo ===")
    
    # Initialize analyzer
    analyzer = VoiceDataAnalyzer()
    
    # Create DataFrame
    print("Creating features DataFrame...")
    df = analyzer.create_features_dataframe(features_list, labels)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary_stats = analyzer.generate_summary_statistics(df)
    
    print("\nSample summary statistics:")
    print(summary_stats.head(10).to_string(index=False))
    
    # Note: Actual plotting requires matplotlib backend
    print("\nNote: To generate visualizations, run:")
    print("analyzer.plot_feature_distributions(df)")
    print("analyzer.plot_correlation_matrix(df)")
    print("analyzer.plot_pca_analysis(df)")
    
    return analyzer, df


def demonstrate_model_evaluation(classifier, X, y):
    """
    Demonstrate model evaluation.
    """
    print("\n=== Model Evaluation Demo ===")
    
    # Evaluate best model
    evaluation = classifier.evaluate_model(classifier.best_model, X, y)
    
    print(f"Best model accuracy: {evaluation['accuracy']:.4f}")
    print(f"Confusion matrix:\n{evaluation['confusion_matrix']}")
    
    # Classification report
    print("\nClassification Report:")
    class_report = evaluation['classification_report']
    for label, metrics in class_report.items():
        if isinstance(metrics, dict):
            print(f"{label}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")


def save_model_example(classifier):
    """
    Demonstrate model saving.
    """
    print("\n=== Model Saving Demo ===")
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, 'gender_classifier.joblib')
    classifier.save_model(model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Demonstrate loading
    new_classifier = GenderClassifier()
    new_classifier.load_model(model_path)
    print("Model loaded successfully!")
    
    return model_path


def main():
    """
    Main function to run the complete demo.
    """
    print("Voice Recording Analysis - Complete Workflow Demo")
    print("=" * 60)
    
    try:
        # Demonstrate each component
        demonstrate_feature_extraction()
        demonstrate_preprocessing()
        
        # Machine learning pipeline
        classifier, X, y, labels, features_list = demonstrate_machine_learning()
        
        # Data analysis
        analyzer, df = demonstrate_data_analysis(classifier, X, y, labels, features_list)
        
        # Model evaluation
        demonstrate_model_evaluation(classifier, X, y)
        
        # Save model
        model_path = save_model_example(classifier)
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("\nNext steps:")
        print("1. Replace sample data with real audio files")
        print("2. Adjust feature extraction parameters for your specific use case")
        print("3. Experiment with different machine learning models")
        print("4. Use the visualization functions to explore your data")
        print(f"5. Load the saved model from: {model_path}")
        
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()