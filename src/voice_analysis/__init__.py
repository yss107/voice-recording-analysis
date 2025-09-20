"""
Voice Recording Analysis Package

A comprehensive toolkit for analyzing voice recordings and predicting gender
based on vocal features such as mean frequency, spectral entropy, and mode frequency.

This package was developed as a take-home assignment solution for data science
positions at Sandvik.
"""

__version__ = "1.0.0"
__author__ = "Voice Analysis Team"

# Try to import audio-related modules, fall back gracefully if dependencies are missing
try:
    from .feature_extraction import VoiceFeatureExtractor
    from .preprocessing import AudioPreprocessor
    _AUDIO_AVAILABLE = True
except ImportError as e:
    VoiceFeatureExtractor = None
    AudioPreprocessor = None
    _AUDIO_AVAILABLE = False
    import warnings
    warnings.warn(f"Audio processing modules not available. Install librosa and soundfile to use audio features. Error: {e}")

# Core ML modules should always be available
from .models import GenderClassifier
from .analysis import VoiceDataAnalyzer

__all__ = [
    "GenderClassifier",
    "VoiceDataAnalyzer"
]

# Add audio modules if available
if _AUDIO_AVAILABLE:
    __all__.extend(["VoiceFeatureExtractor", "AudioPreprocessor"])