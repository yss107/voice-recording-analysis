# Voice Recording Analysis for Gender Classification

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive toolkit for analyzing voice recordings and predicting gender based on vocal features such as mean frequency, spectral entropy, and mode frequency. This project was developed as a take-home assignment solution for data science positions at Sandvik.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for voice-based gender classification, including:

- **Feature Extraction**: Extract 17+ acoustic features from voice recordings
- **Data Preprocessing**: Audio normalization, noise reduction, and data augmentation
- **Machine Learning Models**: Multiple algorithms for gender classification
- **Data Analysis**: Comprehensive visualization and statistical analysis
- **Model Evaluation**: Cross-validation, feature importance, and performance metrics

## 📊 Key Features Extracted

The system extracts comprehensive vocal features including:

### Core Features (from problem statement)
- **Mean Frequency**: Average frequency content of the voice signal
- **Spectral Entropy**: Measure of frequency distribution randomness  
- **Mode Frequency**: Most common frequency in the voice signal

### Additional Features
- **Fundamental Frequency (F0)**: Statistics of pitch
- **Formant Frequencies**: F1, F2, F3 vocal tract resonances
- **Spectral Features**: Centroid, rolloff, bandwidth
- **Temporal Features**: Duration, tempo, energy
- **Voice Quality**: Voiced/unvoiced ratio, zero-crossing rate

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yss107/voice-recording-analysis.git
cd voice-recording-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.voice_analysis import VoiceFeatureExtractor, GenderClassifier

# Extract features from audio file
extractor = VoiceFeatureExtractor()
features = extractor.extract_all_features('path/to/audio/file.wav')

# Train a gender classifier
classifier = GenderClassifier()
# ... (with your training data)

# Make predictions
prediction = classifier.predict(features)
print(f"Predicted gender: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.2f}")
```

### Complete Workflow Example

Run the complete demo:
```bash
python examples/complete_workflow.py
```

### Interactive Analysis

Launch the Jupyter notebook for interactive exploration:
```bash
jupyter notebook notebooks/voice_analysis_exploration.ipynb
```

## 📁 Project Structure

```
voice-recording-analysis/
├── src/voice_analysis/          # Main package
│   ├── __init__.py
│   ├── feature_extraction.py    # Audio feature extraction
│   ├── preprocessing.py         # Audio preprocessing utilities
│   ├── models.py               # Machine learning models
│   └── analysis.py             # Data analysis and visualization
├── examples/                   # Example scripts
│   └── complete_workflow.py    # Complete demo workflow
├── notebooks/                  # Jupyter notebooks
│   └── voice_analysis_exploration.ipynb
├── data/                      # Data directory (for your datasets)
├── models/                    # Saved models
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── README.md
```

## 🎵 Audio Features in Detail

### Frequency Domain Features
- **Mean Frequency**: Weighted average of frequency components
- **Mode Frequency**: Peak frequency with highest energy
- **Spectral Entropy**: Measure of frequency distribution complexity
- **Spectral Centroid**: "Center of mass" of the spectrum
- **Spectral Rolloff**: Frequency below which 85% of energy is contained

### Time Domain Features  
- **Fundamental Frequency (F0)**: Voice pitch statistics
- **Formants**: Vocal tract resonance frequencies (F1, F2, F3)
- **Zero Crossing Rate**: Rate of signal sign changes
- **RMS Energy**: Root mean square energy measure

### Voice Quality Features
- **Voiced Fraction**: Proportion of voiced vs unvoiced segments
- **Spectral Bandwidth**: Width of the frequency spectrum
- **Tempo**: Speech rate estimation

## 🤖 Machine Learning Models

The system includes multiple classification algorithms:

- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble method  
- **Support Vector Machine (SVM)**: Kernel-based classifier
- **Logistic Regression**: Linear probabilistic model
- **K-Nearest Neighbors**: Instance-based learning
- **Multi-layer Perceptron**: Neural network classifier

Each model includes:
- Cross-validation evaluation
- Hyperparameter tuning capabilities
- Feature importance analysis
- Comprehensive performance metrics

## 📈 Data Analysis Features

### Visualization Capabilities
- Feature distribution plots by gender
- Correlation matrix heatmaps
- Principal Component Analysis (PCA)
- Box plots for feature comparison
- Model performance comparisons
- Feature importance rankings

### Statistical Analysis
- Summary statistics by gender
- Feature correlation analysis
- Dimensionality reduction
- Model evaluation metrics

## 🔧 Advanced Usage

### Custom Feature Extraction

```python
# Initialize with custom parameters
extractor = VoiceFeatureExtractor(
    sample_rate=16000,
    frame_length=1024,
    hop_length=256
)

# Extract specific features
mean_freq = extractor.extract_mean_frequency(audio_signal)
entropy = extractor.extract_spectral_entropy(audio_signal)
f0_stats = extractor.extract_fundamental_frequency_stats(audio_signal)
```

### Hyperparameter Tuning

```python
# Tune model hyperparameters
tuning_results = classifier.hyperparameter_tuning(
    X_train, y_train, 
    model_name='random_forest'
)

print(f"Best parameters: {tuning_results['best_params']}")
print(f"Best score: {tuning_results['best_score']:.4f}")
```

### Model Persistence

```python
# Save trained model
classifier.save_model('models/gender_classifier.joblib')

# Load model for inference
new_classifier = GenderClassifier()
new_classifier.load_model('models/gender_classifier.joblib')
```

## 📊 Expected Performance

With well-preprocessed voice data, the system typically achieves:

- **Accuracy**: 85-95% on balanced datasets
- **Cross-validation**: Robust performance across folds
- **Feature Importance**: F0 and formants usually most discriminative
- **Training Time**: Few seconds to minutes depending on dataset size

## 🔬 Research Background

Gender classification from voice is based on physiological differences:

### Male Voices Typically Have:
- Lower fundamental frequency (F0: ~120Hz)
- Lower formant frequencies
- Larger vocal tract (longer formants)
- Different spectral characteristics

### Female Voices Typically Have:
- Higher fundamental frequency (F0: ~200Hz)  
- Higher formant frequencies
- Smaller vocal tract (shorter formants)
- Different energy distribution

## 📝 Data Requirements

### Audio File Formats
- Supported: WAV, MP3, FLAC, M4A, OGG
- Recommended: WAV files at 16kHz or 22kHz
- Duration: 1-10 seconds for optimal results

### Dataset Structure
```
data/
├── male/
│   ├── speaker1_001.wav
│   ├── speaker1_002.wav
│   └── ...
└── female/
    ├── speaker2_001.wav
    ├── speaker2_002.wav
    └── ...
```

Or CSV format:
```csv
file_path,gender
/path/to/audio1.wav,male
/path/to/audio2.wav,female
```

## 🧪 Testing

Run the example workflow to test functionality:
```bash
python examples/complete_workflow.py
```

The example generates synthetic voice data to demonstrate all features when real audio files aren't available.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sandvik**: For providing the interesting take-home assignment
- **LibROSA**: Excellent audio analysis library
- **Scikit-learn**: Comprehensive machine learning toolkit
- **Voice Research Community**: For acoustic feature insights

## 📞 Contact

For questions about this implementation or voice analysis in general, please open an issue in the repository.

---

**Note**: This project demonstrates voice analysis techniques for educational and research purposes. For production use with real voice data, ensure compliance with privacy regulations and obtain appropriate consent from speakers.