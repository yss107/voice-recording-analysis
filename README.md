# Voice Recording Analysis

A comprehensive machine learning project for gender classification using voice recordings and vocal features.

## Overview

This project implements a complete pipeline for analyzing voice recordings to predict gender based on vocal features. It extracts meaningful features from audio files and builds machine learning models to classify speakers as male or female.

## Features

### Audio Feature Extraction
- **Frequency Features**: Mean, median, standard deviation, quantiles, IQR
- **Statistical Features**: Skewness, kurtosis, mode frequency, peak frequency
- **Spectral Features**: Spectral centroid, rolloff, bandwidth, contrast, flatness
- **Temporal Features**: RMS energy, zero-crossing rate, duration
- **MFCC Features**: Mel-frequency cepstral coefficients

### Machine Learning Models
- **Logistic Regression**: Linear classification with regularization
- **Random Forest**: Ensemble method with feature importance
- **Support Vector Machine**: Non-linear classification with RBF kernel

### Analysis & Visualization
- Feature distribution plots by gender
- Correlation matrix analysis
- Model performance comparison
- Feature importance ranking
- Confusion matrices
- Comprehensive reporting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yss107/voice-recording-analysis.git
cd voice-recording-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with Sample Data

Run the analysis with generated sample data:

```bash
python main.py --use-sample-data --save-models --output-dir results
```

### Using Real Audio Data

1. Place your `.wav` audio files in a `data/` directory
2. Organize files with gender information in the path (e.g., `data/male/`, `data/female/`)
3. Run the analysis:

```bash
python main.py --data-dir data --save-models --output-dir results
```

### Command Line Options

- `--data-dir`: Directory containing audio files (default: `data`)
- `--use-sample-data`: Use generated sample data for demonstration
- `--save-models`: Save trained models for later use
- `--output-dir`: Directory for output files (default: `output`)

## Project Structure

```
voice-recording-analysis/
├── voice_analysis/
│   ├── __init__.py
│   ├── data_downloader.py      # Data download and extraction
│   ├── feature_extractor.py    # Audio feature extraction
│   ├── models.py              # Machine learning models
│   └── visualizer.py          # Analysis and visualization
├── main.py                    # Main analysis pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                # Git ignore rules
```

## Output Files

The analysis generates several output files:

- `extracted_features.csv`: Dataset with extracted features
- `plots/`: Directory containing all visualization plots
  - `feature_distributions.png`: Feature distributions by gender
  - `feature_boxplots.png`: Box plots of key features
  - `correlation_matrix.png`: Feature correlation heatmap
  - `feature_importance.png`: Feature importance ranking
  - `model_comparison.png`: Model performance comparison
  - `confusion_matrices.png`: Confusion matrices for all models
- `analysis_report.txt`: Comprehensive text report
- `models/`: Saved trained models (if `--save-models` is used)

## Key Features Analyzed

Based on the vocal analysis literature and the problem statement:

1. **Mean Frequency (kHz)**: Average fundamental frequency
2. **Standard Deviation of Frequency**: Frequency variability
3. **Median Frequency (kHz)**: Middle frequency value
4. **First/Third Quantiles (kHz)**: Frequency distribution quartiles
5. **Inter-Quantile Range (kHz)**: Frequency spread measure
6. **Skewness**: Frequency distribution asymmetry
7. **Kurtosis**: Frequency distribution tail heaviness
8. **Mode Frequency**: Most common frequency
9. **Peak Frequency**: Maximum frequency observed
10. **Spectral Features**: Additional audio characteristics

## Model Performance

The project evaluates models using multiple metrics:

- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity measure
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the receiver operating characteristic curve

## Technical Implementation

### Audio Processing
- Uses `librosa` for robust audio feature extraction
- Handles 16kHz, 16-bit WAV files as specified
- Implements vocal range filtering (human vocal frequencies)

### Machine Learning
- Scikit-learn for model implementation
- Cross-validation for robust evaluation
- Hyperparameter tuning capabilities
- Feature scaling and preprocessing

### Data Handling
- Pandas for data manipulation
- Support for large datasets
- Metadata extraction from audio files

## Example Results

Sample output from the analysis:

```
VOICE RECORDING ANALYSIS SUMMARY REPORT
========================================================

DATASET SUMMARY:
Total samples: 100
Gender distribution:
  male: 50 (50.0%)
  female: 50 (50.0%)
Total features: 31

MODEL PERFORMANCE SUMMARY:
RANDOM_FOREST:
  Accuracy:  0.950
  Precision: 0.951
  Recall:    0.950
  F1-Score:  0.950

BEST PERFORMING MODEL: RANDOM_FOREST
Best Accuracy: 0.950

TOP 10 MOST IMPORTANT FEATURES:
 1. Mean Frequency (kHz): 0.234
 2. Peak Frequency (kHz): 0.187
 3. Median Frequency (kHz): 0.156
...
```

## Enterprise Applications

This voice analysis system can be applied in various enterprise contexts:

- **Voice Authentication**: Biometric security systems
- **Call Center Analytics**: Customer service optimization
- **Healthcare**: Voice-based health monitoring
- **Accessibility**: Voice-controlled interfaces
- **Market Research**: Demographic analysis of voice data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the Sandvik data science assignment requirements
- Uses state-of-the-art audio processing libraries
- Implements best practices for machine learning pipelines