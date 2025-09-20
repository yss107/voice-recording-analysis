#!/usr/bin/env python3
"""
Basic demonstration of voice recording analysis concepts.

This script demonstrates the core concepts of the voice recording analysis
project using only built-in Python libraries.
"""

import random
import math
import statistics
import json

def create_sample_voice_data():
    """
    Create sample voice data that simulates real voice characteristics.
    
    Returns:
        dict: Dictionary containing voice features and labels
    """
    print("Generating sample voice recording data...")
    
    # Set random seed for reproducible results
    random.seed(42)
    
    n_samples_per_gender = 50
    
    # Define voice characteristics based on research
    # Males typically have lower fundamental frequencies (85-180 Hz)
    # Females typically have higher fundamental frequencies (165-265 Hz)
    
    samples = []
    
    # Generate male voice samples
    for i in range(n_samples_per_gender):
        sample = {
            'sample_id': f'male_{i:03d}',
            'gender': 'male',
            'mean_frequency_khz': random.gauss(0.12, 0.02),  # Lower frequency
            'std_frequency_khz': random.gauss(0.05, 0.01),
            'median_frequency_khz': random.gauss(0.11, 0.02),
            'q1_frequency_khz': random.gauss(0.08, 0.015),
            'q3_frequency_khz': random.gauss(0.15, 0.025),
            'iqr_frequency_khz': random.gauss(0.07, 0.015),
            'skewness': random.gauss(0.5, 0.3),
            'kurtosis': random.gauss(2.0, 0.5),
            'mode_frequency_khz': random.gauss(0.10, 0.02),
            'peak_frequency_khz': random.gauss(0.20, 0.04),
            'duration_seconds': random.gauss(2.5, 0.5)
        }
        samples.append(sample)
    
    # Generate female voice samples
    for i in range(n_samples_per_gender):
        sample = {
            'sample_id': f'female_{i:03d}',
            'gender': 'female',
            'mean_frequency_khz': random.gauss(0.18, 0.03),  # Higher frequency
            'std_frequency_khz': random.gauss(0.07, 0.015),
            'median_frequency_khz': random.gauss(0.17, 0.03),
            'q1_frequency_khz': random.gauss(0.13, 0.025),
            'q3_frequency_khz': random.gauss(0.22, 0.035),
            'iqr_frequency_khz': random.gauss(0.09, 0.02),
            'skewness': random.gauss(0.3, 0.4),
            'kurtosis': random.gauss(2.2, 0.6),
            'mode_frequency_khz': random.gauss(0.16, 0.03),
            'peak_frequency_khz': random.gauss(0.30, 0.06),
            'duration_seconds': random.gauss(2.3, 0.4)
        }
        samples.append(sample)
    
    return samples

def analyze_voice_features(samples):
    """
    Analyze voice features and provide statistical insights.
    
    Args:
        samples: List of voice sample dictionaries
    """
    print("\n" + "="*60)
    print("VOICE FEATURE ANALYSIS")
    print("="*60)
    
    # Separate by gender
    male_samples = [s for s in samples if s['gender'] == 'male']
    female_samples = [s for s in samples if s['gender'] == 'female']
    
    print(f"Total samples: {len(samples)}")
    print(f"Male samples: {len(male_samples)}")
    print(f"Female samples: {len(female_samples)}")
    
    # Analyze key features
    features_to_analyze = [
        'mean_frequency_khz',
        'std_frequency_khz', 
        'median_frequency_khz',
        'peak_frequency_khz',
        'skewness',
        'kurtosis'
    ]
    
    print(f"\nAnalyzing {len(features_to_analyze)} key features:")
    print("-" * 60)
    
    for feature in features_to_analyze:
        male_values = [s[feature] for s in male_samples]
        female_values = [s[feature] for s in female_samples]
        
        male_mean = statistics.mean(male_values)
        female_mean = statistics.mean(female_values)
        male_std = statistics.stdev(male_values)
        female_std = statistics.stdev(female_values)
        
        print(f"{feature.upper().replace('_', ' ')}:")
        print(f"  Male:   Mean={male_mean:.4f}, Std={male_std:.4f}")
        print(f"  Female: Mean={female_mean:.4f}, Std={female_std:.4f}")
        print(f"  Difference: {abs(female_mean - male_mean):.4f}")
        print()

def simple_classifier(samples):
    """
    Implement a simple rule-based classifier for gender prediction.
    
    Args:
        samples: List of voice sample dictionaries
        
    Returns:
        dict: Classification results
    """
    print("="*60)
    print("SIMPLE GENDER CLASSIFICATION")
    print("="*60)
    
    # Calculate threshold based on mean frequency
    all_mean_freq = [s['mean_frequency_khz'] for s in samples]
    threshold = statistics.median(all_mean_freq)
    
    print(f"Using mean frequency threshold: {threshold:.4f} kHz")
    print("Rule: If mean_frequency < threshold → Male, else → Female")
    print()
    
    correct_predictions = 0
    total_predictions = len(samples)
    
    confusion_matrix = {'male': {'male': 0, 'female': 0}, 
                       'female': {'male': 0, 'female': 0}}
    
    for sample in samples:
        actual_gender = sample['gender']
        predicted_gender = 'male' if sample['mean_frequency_khz'] < threshold else 'female'
        
        confusion_matrix[actual_gender][predicted_gender] += 1
        
        if actual_gender == predicted_gender:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    
    print("CLASSIFICATION RESULTS:")
    print(f"Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    print()
    
    print("Confusion Matrix:")
    print("                Predicted")
    print("Actual      Male    Female")
    print(f"Male      {confusion_matrix['male']['male']:4d}    {confusion_matrix['male']['female']:4d}")
    print(f"Female    {confusion_matrix['female']['male']:4d}    {confusion_matrix['female']['female']:4d}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'threshold': threshold
    }

def advanced_feature_analysis(samples):
    """
    Perform advanced feature analysis to identify important characteristics.
    
    Args:
        samples: List of voice sample dictionaries
    """
    print("\n" + "="*60)
    print("ADVANCED FEATURE ANALYSIS")
    print("="*60)
    
    # Separate by gender
    male_samples = [s for s in samples if s['gender'] == 'male']
    female_samples = [s for s in samples if s['gender'] == 'female']
    
    # Calculate separability for each feature
    feature_separability = {}
    
    numeric_features = [key for key in samples[0].keys() 
                       if key not in ['sample_id', 'gender'] and 
                       isinstance(samples[0][key], (int, float))]
    
    for feature in numeric_features:
        male_values = [s[feature] for s in male_samples]
        female_values = [s[feature] for s in female_samples]
        
        male_mean = statistics.mean(male_values)
        female_mean = statistics.mean(female_values)
        male_std = statistics.stdev(male_values)
        female_std = statistics.stdev(female_values)
        
        # Calculate Cohen's d (effect size)
        pooled_std = math.sqrt((male_std**2 + female_std**2) / 2)
        cohens_d = abs(male_mean - female_mean) / pooled_std if pooled_std > 0 else 0
        
        feature_separability[feature] = cohens_d
    
    # Sort features by separability
    sorted_features = sorted(feature_separability.items(), 
                           key=lambda x: x[1], reverse=True)
    
    print("FEATURE IMPORTANCE RANKING (Cohen's d):")
    print("Higher values indicate better gender separation")
    print("-" * 60)
    
    for i, (feature, importance) in enumerate(sorted_features, 1):
        interpretation = "Excellent" if importance > 0.8 else \
                        "Large" if importance > 0.5 else \
                        "Medium" if importance > 0.2 else "Small"
        
        print(f"{i:2d}. {feature.replace('_', ' ').title():<25} "
              f"{importance:.3f} ({interpretation})")
    
    return sorted_features

def generate_insights(samples, classification_results, feature_importance):
    """
    Generate insights and recommendations based on the analysis.
    
    Args:
        samples: List of voice sample dictionaries
        classification_results: Results from classification
        feature_importance: Feature importance ranking
    """
    print("\n" + "="*60)
    print("KEY INSIGHTS AND FINDINGS")
    print("="*60)
    
    accuracy = classification_results['accuracy']
    best_features = feature_importance[:3]
    
    print("PERFORMANCE INSIGHTS:")
    print(f"• Achieved {accuracy:.1%} accuracy with simple rule-based classifier")
    print(f"• Used threshold of {classification_results['threshold']:.4f} kHz for mean frequency")
    
    if accuracy > 0.8:
        print("• Excellent performance suggests strong gender-frequency relationship")
    elif accuracy > 0.6:
        print("• Good performance indicates meaningful pattern in voice features")
    else:
        print("• Moderate performance suggests need for more sophisticated features")
    
    print(f"\nFEATURE INSIGHTS:")
    print(f"• Top 3 discriminative features:")
    for i, (feature, importance) in enumerate(best_features, 1):
        print(f"  {i}. {feature.replace('_', ' ').title()} (effect size: {importance:.3f})")
    
    print(f"\nDATA QUALITY INSIGHTS:")
    print(f"• Dataset contains {len(samples)} balanced samples")
    print(f"• {len([k for k in samples[0].keys() if k not in ['sample_id', 'gender']])} features extracted per sample")
    print(f"• Features span frequency, statistical, and temporal domains")
    
    print(f"\nCONCLUSIONS:")
    print(f"• Voice frequency features show significant gender differences")
    print(f"• Mean frequency is the strongest discriminator")
    print(f"• Statistical features (skewness, kurtosis) provide additional value")
    print(f"• Approach demonstrates feasibility of voice-based gender classification")
    
    print(f"\nRECOMMENDations FOR PRODUCTION:")
    print(f"• Implement ensemble methods for improved accuracy")
    print(f"• Add spectral and MFCC features for robustness")
    print(f"• Consider age and language variations in real deployments")
    print(f"• Validate on diverse datasets for generalizability")

def save_results(samples, results, filename="voice_analysis_results.json"):
    """
    Save analysis results to a file.
    
    Args:
        samples: Voice samples data
        results: Analysis results
        filename: Output filename
    """
    output_data = {
        'summary': {
            'total_samples': len(samples),
            'male_samples': len([s for s in samples if s['gender'] == 'male']),
            'female_samples': len([s for s in samples if s['gender'] == 'female']),
            'features_count': len([k for k in samples[0].keys() if k not in ['sample_id', 'gender']]),
            'classification_accuracy': results['accuracy'],
            'threshold_used': results['threshold']
        },
        'confusion_matrix': results['confusion_matrix'],
        'sample_data': samples[:5]  # Save first 5 samples as examples
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {filename}")
    except Exception as e:
        print(f"\n⚠ Could not save results: {e}")

def main():
    """
    Main function that orchestrates the voice recording analysis.
    """
    print("VOICE RECORDING ANALYSIS - DEMONSTRATION")
    print("="*60)
    print("This script demonstrates core concepts of voice-based gender classification")
    print("using simulated voice features based on acoustic research.")
    print()
    
    # Step 1: Generate sample data
    samples = create_sample_voice_data()
    
    # Step 2: Analyze features
    analyze_voice_features(samples)
    
    # Step 3: Perform classification
    classification_results = simple_classifier(samples)
    
    # Step 4: Advanced feature analysis
    feature_importance = advanced_feature_analysis(samples)
    
    # Step 5: Generate insights
    generate_insights(samples, classification_results, feature_importance)
    
    # Step 6: Save results
    save_results(samples, classification_results)
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("✓ Sample voice data generated")
    print("✓ Feature analysis completed")
    print("✓ Gender classification performed")
    print("✓ Feature importance calculated")
    print("✓ Insights and recommendations provided")
    print()
    print("This demonstration shows the complete pipeline for voice recording analysis.")
    print("In a production environment, replace simulated data with real audio feature extraction.")

if __name__ == "__main__":
    main()