"""
Voice Feature Extraction Module

This module provides comprehensive feature extraction capabilities for voice recordings,
including mean frequency, spectral entropy, mode frequency, and other vocal characteristics
useful for gender classification.
"""

import numpy as np
import librosa
import scipy.stats
from typing import Dict, List, Optional, Union
import warnings


class VoiceFeatureExtractor:
    """
    Extracts various acoustic features from voice recordings for analysis and classification.
    
    Features extracted include:
    - Mean frequency
    - Spectral entropy
    - Mode frequency  
    - Fundamental frequency statistics
    - Formant frequencies
    - Spectral characteristics
    - Temporal features
    """
    
    def __init__(self, sample_rate: int = 22050, frame_length: int = 2048, hop_length: int = 512):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_length: Length of FFT window
            hop_length: Number of samples between successive frames
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio file and return normalized audio signal.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Normalized audio signal
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")
    
    def extract_mean_frequency(self, audio: np.ndarray) -> float:
        """
        Calculate the mean frequency of the audio signal.
        
        Args:
            audio: Audio signal
            
        Returns:
            Mean frequency in Hz
        """
        stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Calculate frequency bins
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
        
        # Calculate weighted mean frequency
        mean_freq = np.sum(magnitude * freqs[:, np.newaxis], axis=0) / np.sum(magnitude, axis=0)
        
        # Return mean across time
        return float(np.mean(mean_freq[np.isfinite(mean_freq)]))
    
    def extract_spectral_entropy(self, audio: np.ndarray) -> float:
        """
        Calculate the spectral entropy of the audio signal.
        
        Args:
            audio: Audio signal
            
        Returns:
            Mean spectral entropy
        """
        stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude = np.abs(stft) ** 2
        
        # Normalize to get probability distribution
        magnitude_normalized = magnitude / (np.sum(magnitude, axis=0) + 1e-10)
        
        # Calculate entropy for each frame
        entropy = -np.sum(magnitude_normalized * np.log2(magnitude_normalized + 1e-10), axis=0)
        
        # Return mean entropy across time
        return float(np.mean(entropy[np.isfinite(entropy)]))
    
    def extract_mode_frequency(self, audio: np.ndarray) -> float:
        """
        Calculate the mode frequency (most common frequency) of the audio signal.
        
        Args:
            audio: Audio signal
            
        Returns:
            Mode frequency in Hz
        """
        stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Find the frequency bin with maximum energy for each frame
        max_freq_bins = np.argmax(magnitude, axis=0)
        
        # Convert to frequencies
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
        max_freqs = freqs[max_freq_bins]
        
        # Calculate mode (most common frequency)
        hist, bin_edges = np.histogram(max_freqs, bins=50)
        mode_bin = np.argmax(hist)
        mode_freq = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
        
        return float(mode_freq)
    
    def extract_fundamental_frequency_stats(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract fundamental frequency (F0) statistics.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with F0 statistics
        """
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            # Filter out unvoiced segments
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) == 0:
                return {
                    'f0_mean': 0.0,
                    'f0_std': 0.0,
                    'f0_min': 0.0,
                    'f0_max': 0.0,
                    'f0_range': 0.0,
                    'voiced_fraction': 0.0
                }
            
            return {
                'f0_mean': float(np.mean(f0_voiced)),
                'f0_std': float(np.std(f0_voiced)),
                'f0_min': float(np.min(f0_voiced)),
                'f0_max': float(np.max(f0_voiced)),
                'f0_range': float(np.max(f0_voiced) - np.min(f0_voiced)),
                'voiced_fraction': float(np.sum(voiced_flag) / len(voiced_flag))
            }
        except:
            # Fallback if pyin fails
            return {
                'f0_mean': 0.0,
                'f0_std': 0.0,
                'f0_min': 0.0,
                'f0_max': 0.0,
                'f0_range': 0.0,
                'voiced_fraction': 0.0
            }
    
    def extract_formant_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract formant frequency features (simplified estimation).
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with formant features
        """
        # Use LPC to estimate formants (simplified approach)
        try:
            # Pre-emphasize the signal
            pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
            
            # Window the signal
            windowed = pre_emphasized * np.hamming(len(pre_emphasized))
            
            # Calculate LPC coefficients
            from scipy.signal import lfilter
            
            # Simple formant estimation using spectral peaks
            stft = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            mean_spectrum = np.mean(magnitude, axis=1)
            
            # Find peaks in the spectrum (simplified formant estimation)
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            
            # Focus on speech frequency range (300-3400 Hz)
            speech_mask = (freqs >= 300) & (freqs <= 3400)
            speech_spectrum = mean_spectrum[speech_mask]
            speech_freqs = freqs[speech_mask]
            
            # Find the top 3 peaks as formant estimates
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(speech_spectrum, height=np.max(speech_spectrum) * 0.1)
            
            if len(peaks) >= 2:
                peak_freqs = speech_freqs[peaks]
                sorted_peaks = np.sort(peak_freqs)
                
                return {
                    'formant_f1': float(sorted_peaks[0]) if len(sorted_peaks) > 0 else 0.0,
                    'formant_f2': float(sorted_peaks[1]) if len(sorted_peaks) > 1 else 0.0,
                    'formant_f3': float(sorted_peaks[2]) if len(sorted_peaks) > 2 else 0.0,
                }
            else:
                return {'formant_f1': 0.0, 'formant_f2': 0.0, 'formant_f3': 0.0}
                
        except:
            return {'formant_f1': 0.0, 'formant_f2': 0.0, 'formant_f3': 0.0}
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract additional spectral features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with spectral features
        """
        # Calculate spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_std': float(np.std(spectral_rolloff)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
            'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
            'zero_crossing_rate_std': float(np.std(zero_crossing_rate)),
        }
    
    def extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features from the audio signal.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with temporal features
        """
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        
        # Calculate tempo and beat features
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            tempo = float(tempo)
        except:
            tempo = 0.0
        
        return {
            'duration': float(len(audio) / self.sample_rate),
            'rms_energy_mean': float(np.mean(rms)),
            'rms_energy_std': float(np.std(rms)),
            'tempo': tempo,
        }
    
    def extract_all_features(self, audio_path_or_signal: Union[str, np.ndarray]) -> Dict[str, float]:
        """
        Extract all available features from an audio file or signal.
        
        Args:
            audio_path_or_signal: Path to audio file or audio signal array
            
        Returns:
            Dictionary containing all extracted features
        """
        # Load audio if path is provided
        if isinstance(audio_path_or_signal, str):
            audio = self.load_audio(audio_path_or_signal)
        else:
            audio = audio_path_or_signal
        
        features = {}
        
        # Extract core features mentioned in the problem statement
        features['mean_frequency'] = self.extract_mean_frequency(audio)
        features['spectral_entropy'] = self.extract_spectral_entropy(audio)
        features['mode_frequency'] = self.extract_mode_frequency(audio)
        
        # Extract additional features
        features.update(self.extract_fundamental_frequency_stats(audio))
        features.update(self.extract_formant_features(audio))
        features.update(self.extract_spectral_features(audio))
        features.update(self.extract_temporal_features(audio))
        
        return features
    
    def extract_features_batch(self, audio_paths: List[str]) -> List[Dict[str, float]]:
        """
        Extract features from multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for i, path in enumerate(audio_paths):
            try:
                features = self.extract_all_features(path)
                features['file_path'] = path
                features_list.append(features)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(audio_paths)} files")
                    
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                continue
        
        return features_list