"""
Real Audio Processing Example

This script demonstrates how to use the voice analysis package with real audio files.
It shows the complete pipeline from audio files to gender predictions.

Requirements:
- pip install librosa soundfile
- Audio files in data/male/ and data/female/ directories
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from voice_analysis import VoiceFeatureExtractor, AudioPreprocessor, GenderClassifier, VoiceDataAnalyzer
    AUDIO_AVAILABLE = True
except ImportError as e:
    print(f"Audio processing not available: {e}")
    print("Install with: pip install librosa soundfile")
    AUDIO_AVAILABLE = False
    sys.exit(1)


def process_audio_dataset(data_dir):
    """
    Process a dataset of audio files and extract features.
    
    Expected directory structure:
    data_dir/
    ├── male/
    │   ├── speaker1_001.wav
    │   └── speaker1_002.wav
    └── female/
        ├── speaker2_001.wav
        └── speaker2_002.wav
    """
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        print("Please create the directory and add audio files.")
        return None, None
    
    print(f"Processing audio files from {data_dir}...")
    
    # Initialize components
    preprocessor = AudioPreprocessor(sample_rate=22050, target_length=3.0)
    extractor = VoiceFeatureExtractor(sample_rate=22050)
    
    # Create dataset from directory structure
    file_paths, labels = preprocessor.create_dataset_from_directory(data_dir)
    
    if not file_paths:
        print("No audio files found in the directory.")
        return None, None
    
    print(f"Found {len(file_paths)} audio files")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Validate dataset
    stats = preprocessor.validate_dataset(file_paths, labels)
    print(f"Valid files: {stats['valid_files']}/{stats['total_files']}")
    
    if stats['valid_files'] == 0:
        print("No valid audio files found.")
        return None, None
    
    # Extract features from all files
    print("Extracting features from audio files...")
    features_list = []
    
    for i, file_path in enumerate(file_paths):
        try:
            # Preprocess audio
            audio = preprocessor.load_and_preprocess(file_path)
            
            # Extract features
            features = extractor.extract_all_features(audio)
            features['file_path'] = file_path
            features['label'] = labels[i]
            
            features_list.append(features)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(file_paths)} files")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Successfully extracted features from {len(features_list)} files")
    return features_list, [f['label'] for f in features_list]


def train_and_evaluate_model(features_list, labels):
    """Train and evaluate gender classification models."""
    print("\nTraining gender classification models...")
    
    # Initialize classifier
    classifier = GenderClassifier(random_state=42)
    
    # Prepare features
    X, feature_names = classifier.prepare_features(features_list)
    y = np.array(labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Train models
    results = classifier.train_all_models(X, y)
    
    # Display results
    print("\nModel Performance:")
    for model_name, model_results in results.items():
        cv_mean = model_results['cv_mean']
        cv_std = model_results['cv_std']
        print(f"{model_name:20}: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Feature importance
    try:
        importance = classifier.get_feature_importance()
        print("\nTop 10 Most Important Features:")
        for i, (feature, score) in enumerate(list(importance.items())[:10]):
            print(f"{i+1:2d}. {feature.replace('_', ' ').title()}: {score:.4f}")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
    
    return classifier


def analyze_data(features_list, labels):
    """Perform data analysis and visualization."""
    print("\nPerforming data analysis...")
    
    # Initialize analyzer
    analyzer = VoiceDataAnalyzer()
    
    # Create DataFrame
    df = analyzer.create_features_dataframe(features_list, labels)
    
    # Summary statistics
    summary = analyzer.generate_summary_statistics(df)
    print(f"Generated summary statistics for {len(summary)} feature-gender combinations")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save summary statistics
    summary_path = os.path.join(results_dir, 'feature_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    # Save feature data
    features_path = os.path.join(results_dir, 'extracted_features.csv')
    df.to_csv(features_path, index=False)
    print(f"Feature data saved to: {features_path}")
    
    # Generate plots (optional, requires display)
    try:
        print("Generating visualizations...")
        
        # Feature distributions
        key_features = ['mean_frequency', 'spectral_entropy', 'mode_frequency', 'f0_mean']
        analyzer.plot_feature_distributions(
            df, features=key_features,
            save_path=os.path.join(results_dir, 'feature_distributions.png')
        )
        
        # Correlation matrix
        analyzer.plot_correlation_matrix(
            df, save_path=os.path.join(results_dir, 'correlation_matrix.png')
        )
        
        # PCA analysis
        analyzer.plot_pca_analysis(
            df, save_path=os.path.join(results_dir, 'pca_analysis.png')
        )
        
        print("Visualizations saved to results/ directory")
        
    except Exception as e:
        print(f"Could not generate visualizations: {e}")
    
    return df


def main():
    """Main function to run the real audio processing example."""
    print("Voice Recording Analysis - Real Audio Processing Example")
    print("=" * 60)
    
    if not AUDIO_AVAILABLE:
        return
    
    # Set data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    print(f"Looking for audio files in: {data_dir}")
    print("Expected structure:")
    print("data/")
    print("├── male/")
    print("│   ├── audio1.wav")
    print("│   └── audio2.wav")
    print("└── female/")
    print("    ├── audio3.wav")
    print("    └── audio4.wav")
    print()
    
    # Process audio dataset
    features_list, labels = process_audio_dataset(data_dir)
    
    if features_list is None:
        print("No audio data found. Creating example directory structure...")
        
        # Create example directory structure
        os.makedirs(os.path.join(data_dir, 'male'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'female'), exist_ok=True)
        
        # Create README
        readme_path = os.path.join(data_dir, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write("Audio Data Directory\n")
            f.write("===================\n\n")
            f.write("Place your audio files here:\n\n")
            f.write("male/\n")
            f.write("  - Place male voice recordings here (WAV, MP3, etc.)\n\n")
            f.write("female/\n")
            f.write("  - Place female voice recordings here (WAV, MP3, etc.)\n\n")
            f.write("Supported formats: WAV, MP3, FLAC, M4A, OGG\n")
            f.write("Recommended: WAV files at 16kHz or 22kHz sample rate\n")
        
        print(f"Created directory structure at: {data_dir}")
        print(f"Please add audio files and run again.")
        return
    
    # Train models
    classifier = train_and_evaluate_model(features_list, labels)
    
    # Analyze data
    df = analyze_data(features_list, labels)
    
    # Save model
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'real_audio_classifier.joblib')
    classifier.save_model(model_path)
    print(f"\nTrained model saved to: {model_path}")
    
    # Test prediction on first sample
    if features_list:
        sample_features = features_list[0]
        prediction = classifier.predict(sample_features)
        print(f"\nSample prediction:")
        print(f"File: {sample_features.get('file_path', 'Unknown')}")
        print(f"Actual: {sample_features.get('label', 'Unknown')}")
        print(f"Predicted: {prediction['prediction']}")
        if 'confidence' in prediction:
            print(f"Confidence: {prediction['confidence']:.4f}")
    
    print("\n" + "=" * 60)
    print("Real audio processing completed successfully!")
    print("\nResults saved in:")
    print(f"- Feature data: results/extracted_features.csv")
    print(f"- Summary stats: results/feature_summary.csv")
    print(f"- Trained model: models/real_audio_classifier.joblib")
    print(f"- Visualizations: results/*.png")


if __name__ == "__main__":
    main()