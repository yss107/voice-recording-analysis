"""
Audio Preprocessing Module

This module provides utilities for preprocessing audio data, including
noise reduction, normalization, and data augmentation techniques.
"""

import numpy as np
import librosa
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import os
from pathlib import Path


class AudioPreprocessor:
    """
    Handles preprocessing of audio data for voice analysis.
    
    Includes functionality for:
    - Audio loading and normalization
    - Noise reduction
    - Data augmentation
    - Dataset organization
    """
    
    def __init__(self, sample_rate: int = 22050, target_length: Optional[float] = None):
        """
        Initialize the audio preprocessor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            target_length: Target length in seconds (None for variable length)
        """
        self.sample_rate = sample_rate
        self.target_length = target_length
        
    def load_and_preprocess(self, file_path: str, normalize: bool = True) -> np.ndarray:
        """
        Load and preprocess an audio file.
        
        Args:
            file_path: Path to audio file
            normalize: Whether to normalize the audio
            
        Returns:
            Preprocessed audio signal
        """
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Normalize if requested
            if normalize:
                audio = self.normalize_audio(audio)
            
            # Trim silence
            audio = self.trim_silence(audio)
            
            # Adjust length if target length is specified
            if self.target_length is not None:
                audio = self.adjust_length(audio, self.target_length)
            
            return audio
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess audio file {file_path}: {str(e)}")
    
    def normalize_audio(self, audio: np.ndarray, method: str = 'peak') -> np.ndarray:
        """
        Normalize audio signal.
        
        Args:
            audio: Audio signal
            method: Normalization method ('peak', 'rms')
            
        Returns:
            Normalized audio signal
        """
        if method == 'peak':
            # Peak normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
        elif method == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio / rms
        
        return audio
    
    def trim_silence(self, audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio.
        
        Args:
            audio: Audio signal
            frame_length: Length of analysis frames
            hop_length: Hop length between frames
            
        Returns:
            Trimmed audio signal
        """
        # Use librosa's trim function
        audio_trimmed, _ = librosa.effects.trim(
            audio, 
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        return audio_trimmed
    
    def adjust_length(self, audio: np.ndarray, target_length: float) -> np.ndarray:
        """
        Adjust audio length to target duration.
        
        Args:
            audio: Audio signal
            target_length: Target length in seconds
            
        Returns:
            Length-adjusted audio signal
        """
        target_samples = int(target_length * self.sample_rate)
        current_samples = len(audio)
        
        if current_samples > target_samples:
            # Truncate if too long
            return audio[:target_samples]
        elif current_samples < target_samples:
            # Pad if too short
            padding = target_samples - current_samples
            return np.pad(audio, (0, padding), mode='constant', constant_values=0)
        else:
            return audio
    
    def apply_pre_emphasis(self, audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """
        Apply pre-emphasis filter to audio.
        
        Args:
            audio: Audio signal
            alpha: Pre-emphasis coefficient
            
        Returns:
            Pre-emphasized audio signal
        """
        return np.append(audio[0], audio[1:] - alpha * audio[:-1])
    
    def reduce_noise(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        Simple noise reduction using spectral subtraction.
        
        Args:
            audio: Audio signal
            noise_factor: Factor for noise reduction
            
        Returns:
            Noise-reduced audio signal
        """
        # Estimate noise from the first 0.5 seconds
        noise_sample_length = min(len(audio), int(0.5 * self.sample_rate))
        noise_spectrum = np.abs(librosa.stft(audio[:noise_sample_length]))
        noise_profile = np.mean(noise_spectrum, axis=1, keepdims=True)
        
        # Apply spectral subtraction
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Subtract noise
        clean_magnitude = magnitude - noise_factor * noise_profile
        clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
        
        # Reconstruct audio
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft)
        
        return clean_audio
    
    def augment_audio(self, audio: np.ndarray, augmentation_type: str) -> np.ndarray:
        """
        Apply data augmentation to audio.
        
        Args:
            audio: Audio signal
            augmentation_type: Type of augmentation ('pitch_shift', 'time_stretch', 'noise')
            
        Returns:
            Augmented audio signal
        """
        if augmentation_type == 'pitch_shift':
            # Random pitch shift
            n_steps = np.random.uniform(-2, 2)
            return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        
        elif augmentation_type == 'time_stretch':
            # Random time stretching
            rate = np.random.uniform(0.8, 1.2)
            return librosa.effects.time_stretch(audio, rate=rate)
        
        elif augmentation_type == 'noise':
            # Add random noise
            noise_factor = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_factor, len(audio))
            return audio + noise
        
        else:
            return audio
    
    def create_dataset_from_directory(self, 
                                    data_dir: str, 
                                    label_mapping: Optional[Dict[str, str]] = None) -> Tuple[List[str], List[str]]:
        """
        Create dataset from directory structure.
        
        Expected structure:
        data_dir/
        ├── male/
        │   ├── file1.wav
        │   └── file2.wav
        └── female/
            ├── file3.wav
            └── file4.wav
        
        Args:
            data_dir: Root directory containing audio files
            label_mapping: Optional mapping from folder names to labels
            
        Returns:
            Tuple of (file_paths, labels)
        """
        if label_mapping is None:
            label_mapping = {'male': 'male', 'female': 'female', 'man': 'male', 'woman': 'female'}
        
        file_paths = []
        labels = []
        
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")
        
        # Audio file extensions to look for
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        
        for subfolder in data_path.iterdir():
            if subfolder.is_dir():
                folder_name = subfolder.name.lower()
                
                # Map folder name to label
                label = label_mapping.get(folder_name, folder_name)
                
                # Find audio files in this folder
                for audio_file in subfolder.iterdir():
                    if audio_file.suffix.lower() in audio_extensions:
                        file_paths.append(str(audio_file))
                        labels.append(label)
        
        return file_paths, labels
    
    def create_dataset_from_csv(self, csv_path: str, 
                               audio_column: str = 'file_path',
                               label_column: str = 'gender') -> Tuple[List[str], List[str]]:
        """
        Create dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            audio_column: Column name containing audio file paths
            label_column: Column name containing labels
            
        Returns:
            Tuple of (file_paths, labels)
        """
        df = pd.read_csv(csv_path)
        
        if audio_column not in df.columns:
            raise ValueError(f"Column '{audio_column}' not found in CSV")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in CSV")
        
        file_paths = df[audio_column].tolist()
        labels = df[label_column].tolist()
        
        return file_paths, labels
    
    def validate_dataset(self, file_paths: List[str], labels: List[str]) -> Dict[str, any]:
        """
        Validate dataset and return statistics.
        
        Args:
            file_paths: List of audio file paths
            labels: List of corresponding labels
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_files': len(file_paths),
            'valid_files': 0,
            'invalid_files': 0,
            'label_distribution': {},
            'duration_stats': [],
            'sample_rate_stats': []
        }
        
        # Count label distribution
        for label in labels:
            stats['label_distribution'][label] = stats['label_distribution'].get(label, 0) + 1
        
        # Validate files
        for i, file_path in enumerate(file_paths):
            try:
                if os.path.exists(file_path):
                    # Try to load audio and get basic info
                    audio, sr = librosa.load(file_path, sr=None)
                    duration = len(audio) / sr
                    
                    stats['valid_files'] += 1
                    stats['duration_stats'].append(duration)
                    stats['sample_rate_stats'].append(sr)
                else:
                    stats['invalid_files'] += 1
                    print(f"File not found: {file_path}")
                    
            except Exception as e:
                stats['invalid_files'] += 1
                print(f"Error loading {file_path}: {str(e)}")
        
        # Calculate duration statistics
        if stats['duration_stats']:
            stats['mean_duration'] = np.mean(stats['duration_stats'])
            stats['std_duration'] = np.std(stats['duration_stats'])
            stats['min_duration'] = np.min(stats['duration_stats'])
            stats['max_duration'] = np.max(stats['duration_stats'])
        
        # Calculate sample rate statistics
        if stats['sample_rate_stats']:
            stats['unique_sample_rates'] = list(set(stats['sample_rate_stats']))
        
        return stats
    
    def create_train_val_split(self, 
                             file_paths: List[str], 
                             labels: List[str], 
                             val_ratio: float = 0.2,
                             stratify: bool = True,
                             random_state: int = 42) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Create train/validation split.
        
        Args:
            file_paths: List of audio file paths
            labels: List of corresponding labels
            val_ratio: Fraction of data to use for validation
            stratify: Whether to stratify the split by labels
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        from sklearn.model_selection import train_test_split
        
        if stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                file_paths, labels,
                test_size=val_ratio,
                stratify=labels,
                random_state=random_state
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                file_paths, labels,
                test_size=val_ratio,
                random_state=random_state
            )
        
        return X_train, X_val, y_train, y_val