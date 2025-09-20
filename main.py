#!/usr/bin/env python3
"""
Main analysis script for voice recording gender classification.

This script demonstrates the complete pipeline for voice recording analysis,
from data preparation to model evaluation and visualization.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse

# Add the voice_analysis module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voice_analysis.data_downloader import VoiceDataDownloader
from voice_analysis.feature_extractor import VoiceFeatureExtractor
from voice_analysis.models import VoiceGenderClassifier
from voice_analysis.visualizer import VoiceAnalysisVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """
    Create sample data for demonstration purposes.
    
    In a real-world scenario, this would be replaced with actual
    data download and processing from the voice recording repository.
    """
    logger.info("Creating sample dataset for demonstration...")
    
    # Create sample features that mimic real voice characteristics
    np.random.seed(42)
    
    # Simulate data for 100 samples (50 male, 50 female)
    n_samples_per_gender = 50
    
    # Male voice characteristics (typically lower frequencies)
    male_data = {
        'mean_frequency_khz': np.random.normal(0.12, 0.02, n_samples_per_gender),
        'std_frequency_khz': np.random.normal(0.05, 0.01, n_samples_per_gender),
        'median_frequency_khz': np.random.normal(0.11, 0.02, n_samples_per_gender),
        'q1_frequency_khz': np.random.normal(0.08, 0.015, n_samples_per_gender),
        'q3_frequency_khz': np.random.normal(0.15, 0.025, n_samples_per_gender),
        'iqr_frequency_khz': np.random.normal(0.07, 0.015, n_samples_per_gender),
        'skewness': np.random.normal(0.5, 0.3, n_samples_per_gender),
        'kurtosis': np.random.normal(2.0, 0.5, n_samples_per_gender),
        'mode_frequency_khz': np.random.normal(0.10, 0.02, n_samples_per_gender),
        'peak_frequency_khz': np.random.normal(0.20, 0.04, n_samples_per_gender),
        'mean_spectral_rolloff_khz': np.random.normal(0.25, 0.05, n_samples_per_gender),
        'mean_spectral_bandwidth_khz': np.random.normal(0.08, 0.02, n_samples_per_gender),
        'mean_rms': np.random.normal(0.1, 0.02, n_samples_per_gender),
        'std_rms': np.random.normal(0.05, 0.01, n_samples_per_gender),
        'mean_zcr': np.random.normal(0.05, 0.01, n_samples_per_gender),
        'std_zcr': np.random.normal(0.02, 0.005, n_samples_per_gender),
        'duration_seconds': np.random.normal(2.5, 0.5, n_samples_per_gender),
        'label': ['male'] * n_samples_per_gender
    }
    
    # Female voice characteristics (typically higher frequencies)
    female_data = {
        'mean_frequency_khz': np.random.normal(0.18, 0.03, n_samples_per_gender),
        'std_frequency_khz': np.random.normal(0.07, 0.015, n_samples_per_gender),
        'median_frequency_khz': np.random.normal(0.17, 0.03, n_samples_per_gender),
        'q1_frequency_khz': np.random.normal(0.13, 0.025, n_samples_per_gender),
        'q3_frequency_khz': np.random.normal(0.22, 0.035, n_samples_per_gender),
        'iqr_frequency_khz': np.random.normal(0.09, 0.02, n_samples_per_gender),
        'skewness': np.random.normal(0.3, 0.4, n_samples_per_gender),
        'kurtosis': np.random.normal(2.2, 0.6, n_samples_per_gender),
        'mode_frequency_khz': np.random.normal(0.16, 0.03, n_samples_per_gender),
        'peak_frequency_khz': np.random.normal(0.30, 0.06, n_samples_per_gender),
        'mean_spectral_rolloff_khz': np.random.normal(0.35, 0.07, n_samples_per_gender),
        'mean_spectral_bandwidth_khz': np.random.normal(0.10, 0.025, n_samples_per_gender),
        'mean_rms': np.random.normal(0.08, 0.015, n_samples_per_gender),
        'std_rms': np.random.normal(0.04, 0.008, n_samples_per_gender),
        'mean_zcr': np.random.normal(0.07, 0.015, n_samples_per_gender),
        'std_zcr': np.random.normal(0.025, 0.008, n_samples_per_gender),
        'duration_seconds': np.random.normal(2.3, 0.4, n_samples_per_gender),
        'label': ['female'] * n_samples_per_gender
    }
    
    # Combine male and female data
    combined_data = {}
    for key in male_data.keys():
        if key == 'label':
            combined_data[key] = male_data[key] + female_data[key]
        else:
            combined_data[key] = np.concatenate([male_data[key], female_data[key]])
    
    # Create DataFrame
    df = pd.DataFrame(combined_data)
    
    # Add some MFCC features for completeness
    for i in range(1, 6):  # First 5 MFCC features
        df[f'mfcc_{i}_mean'] = np.random.normal(0, 1, len(df))
        df[f'mfcc_{i}_std'] = np.random.normal(0.5, 0.2, len(df))
    
    # Add file information
    df['file_name'] = [f'sample_{i:03d}.wav' for i in range(len(df))]
    df['file_path'] = [f'/data/sample_{i:03d}.wav' for i in range(len(df))]
    
    logger.info(f"Created sample dataset with {len(df)} samples")
    return df


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description='Voice Recording Analysis')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--use-sample-data', action='store_true', 
                       help='Use generated sample data instead of real data')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Starting Voice Recording Analysis Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Data preparation
    logger.info("Step 1: Data Preparation")
    
    if args.use_sample_data:
        # Use sample data for demonstration
        df = create_sample_data()
    else:
        # Initialize data downloader
        downloader = VoiceDataDownloader(args.data_dir)
        
        # Download sample data (placeholder implementation)
        downloader.download_sample_data()
        
        # Initialize feature extractor
        extractor = VoiceFeatureExtractor()
        
        # Get audio files
        wav_files = downloader.list_wav_files(args.data_dir)
        
        if not wav_files:
            logger.warning("No .wav files found. Using sample data instead.")
            df = create_sample_data()
        else:
            # Extract features from audio files
            logger.info(f"Found {len(wav_files)} audio files")
            
            # Create labels based on file paths (this would need to be adapted
            # based on actual dataset structure)
            labels = []
            for file_path in wav_files:
                if 'male' in file_path.lower():
                    labels.append('male')
                elif 'female' in file_path.lower():
                    labels.append('female')
                else:
                    labels.append('unknown')
            
            # Extract features
            df = extractor.process_audio_files(wav_files, labels)
    
    # Save the feature dataset
    df.to_csv(output_dir / 'extracted_features.csv', index=False)
    logger.info(f"Feature dataset saved to {output_dir / 'extracted_features.csv'}")
    
    # Step 2: Data exploration and visualization
    logger.info("\nStep 2: Data Exploration and Visualization")
    
    # Initialize visualizer
    visualizer = VoiceAnalysisVisualizer(save_dir=output_dir / 'plots')
    
    # Basic dataset statistics
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features extracted: {len(df.select_dtypes(include=[np.number]).columns)}")
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        logger.info("Gender distribution:")
        for gender, count in label_counts.items():
            logger.info(f"  {gender}: {count}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    visualizer.plot_feature_distributions(df)
    visualizer.plot_feature_boxplots(df)
    visualizer.plot_correlation_matrix(df)
    
    # Step 3: Model training and evaluation
    logger.info("\nStep 3: Model Training and Evaluation")
    
    # Initialize classifier
    classifier = VoiceGenderClassifier(random_state=42)
    
    # Prepare data
    X, y = classifier.prepare_data(df, target_column='label')
    
    # Split data
    X_train, X_test, y_train, y_test = classifier.split_data(X, y, test_size=0.2)
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Train models
    logger.info("Training models...")
    cv_scores = classifier.train_models(X_train, y_train)
    
    # Display cross-validation results
    logger.info("\nCross-validation Results:")
    for model_name, scores in cv_scores.items():
        logger.info(f"{model_name}: {scores['mean_cv_score']:.3f} (+/- {scores['std_cv_score']:.3f})")
    
    # Evaluate models
    logger.info("\nEvaluating models on test set...")
    results = classifier.evaluate_models(X_test, y_test)
    
    # Display test results
    logger.info("\nTest Set Results:")
    for model_name, result in results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Accuracy: {result['accuracy']:.3f}")
        logger.info(f"  Precision: {result['precision']:.3f}")
        logger.info(f"  Recall: {result['recall']:.3f}")
        logger.info(f"  F1-Score: {result['f1_score']:.3f}")
    
    # Step 4: Feature importance and model analysis
    logger.info("\nStep 4: Feature Importance and Model Analysis")
    
    # Get feature importance from Random Forest
    feature_importance = classifier.get_feature_importance('random_forest')
    
    if feature_importance:
        logger.info("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
            logger.info(f"{i+1:2d}. {feature}: {importance:.3f}")
        
        # Plot feature importance
        visualizer.plot_feature_importance(feature_importance)
    
    # Step 5: Model comparison and final visualizations
    logger.info("\nStep 5: Model Comparison and Visualization")
    
    # Create comparison plots
    visualizer.plot_model_comparison(results)
    
    # Create confusion matrices
    class_names = list(classifier.label_encoder.classes_)
    visualizer.plot_confusion_matrices(results, class_names)
    
    # Generate summary report
    report = visualizer.create_summary_report(df, results, feature_importance)
    print("\n" + report)
    
    # Step 6: Save models (if requested)
    if args.save_models:
        logger.info("\nStep 6: Saving Models")
        classifier.save_models(output_dir / 'models')
        logger.info("Models saved successfully")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Files generated:")
    logger.info("  - extracted_features.csv")
    logger.info("  - plots/")
    logger.info("  - analysis_report.txt")
    if args.save_models:
        logger.info("  - models/")
    
    # Best model recommendation
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    logger.info(f"\nRecommended model: {best_model} (Accuracy: {best_accuracy:.3f})")


if __name__ == "__main__":
    main()