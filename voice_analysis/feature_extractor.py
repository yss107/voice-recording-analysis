"""
Feature extraction module for voice recording analysis.

This module extracts vocal features from audio files that can be used
for gender prediction and analysis.
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy.stats import skew, kurtosis

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceFeatureExtractor:
    """Extracts features from voice recordings for analysis."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio files (default: 16000 Hz)
        """
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            return np.array([]), 0
    
    def extract_frequency_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract frequency-based features from audio data.
        
        Args:
            audio: Audio time series
            sr: Sample rate
            
        Returns:
            Dict[str, float]: Dictionary of extracted features
        """
        features = {}
        
        # Compute spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        # Convert to frequency (Hz to kHz)
        spectral_centroids_khz = spectral_centroids / 1000
        spectral_rolloff_khz = spectral_rolloff / 1000
        spectral_bandwidth_khz = spectral_bandwidth / 1000
        
        # Statistical measures
        features['mean_frequency_khz'] = np.mean(spectral_centroids_khz)
        features['std_frequency_khz'] = np.std(spectral_centroids_khz)
        features['median_frequency_khz'] = np.median(spectral_centroids_khz)
        features['q1_frequency_khz'] = np.percentile(spectral_centroids_khz, 25)
        features['q3_frequency_khz'] = np.percentile(spectral_centroids_khz, 75)
        features['iqr_frequency_khz'] = features['q3_frequency_khz'] - features['q1_frequency_khz']
        features['skewness'] = skew(spectral_centroids_khz)
        features['kurtosis'] = kurtosis(spectral_centroids_khz)
        
        # Mode frequency (most common frequency bin)
        hist, bin_edges = np.histogram(spectral_centroids_khz, bins=50)
        mode_idx = np.argmax(hist)
        features['mode_frequency_khz'] = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
        
        # Peak frequency (maximum frequency)
        features['peak_frequency_khz'] = np.max(spectral_centroids_khz)
        
        # Additional features
        features['mean_spectral_rolloff_khz'] = np.mean(spectral_rolloff_khz)
        features['mean_spectral_bandwidth_khz'] = np.mean(spectral_bandwidth_khz)
        
        return features
    
    def extract_temporal_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract temporal features from audio data.
        
        Args:
            audio: Audio time series
            sr: Sample rate
            
        Returns:
            Dict[str, float]: Dictionary of extracted features
        """
        features = {}
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features['mean_rms'] = np.mean(rms)
        features['std_rms'] = np.std(rms)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['mean_zcr'] = np.mean(zcr)
        features['std_zcr'] = np.std(zcr)
        
        # Duration
        features['duration_seconds'] = len(audio) / sr
        
        return features
    
    def extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract spectral features from audio data.
        
        Args:
            audio: Audio time series
            sr: Sample rate
            
        Returns:
            Dict[str, float]: Dictionary of extracted features
        """
        features = {}
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['mean_spectral_contrast'] = np.mean(spectral_contrast)
        features['std_spectral_contrast'] = np.std(spectral_contrast)
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        features['mean_spectral_flatness'] = np.mean(spectral_flatness)
        
        # MFCC features (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        return features
    
    def extract_all_features(self, file_path: str) -> Dict[str, float]:
        """
        Extract all features from an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict[str, float]: Dictionary of all extracted features
        """
        audio, sr = self.load_audio(file_path)
        
        if len(audio) == 0:
            return {}
        
        # Extract different types of features
        freq_features = self.extract_frequency_features(audio, sr)
        temporal_features = self.extract_temporal_features(audio, sr)
        spectral_features = self.extract_spectral_features(audio, sr)
        
        # Combine all features
        all_features = {**freq_features, **temporal_features, **spectral_features}
        
        # Add file information
        all_features['file_path'] = str(file_path)
        all_features['file_name'] = Path(file_path).name
        
        return all_features
    
    def process_audio_files(self, file_paths: List[str], 
                          labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process multiple audio files and extract features.
        
        Args:
            file_paths: List of audio file paths
            labels: Optional list of labels for the audio files
            
        Returns:
            pd.DataFrame: DataFrame containing features for all files
        """
        feature_list = []
        
        for i, file_path in enumerate(file_paths):
            logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
            
            features = self.extract_all_features(file_path)
            
            if features:  # Only add if features were successfully extracted
                if labels:
                    features['label'] = labels[i] if i < len(labels) else 'unknown'
                feature_list.append(features)
        
        if not feature_list:
            logger.warning("No features were successfully extracted")
            return pd.DataFrame()
        
        df = pd.DataFrame(feature_list)
        logger.info(f"Successfully processed {len(df)} audio files")
        
        return df