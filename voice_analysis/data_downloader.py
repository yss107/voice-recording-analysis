"""
Data download module for voice recording analysis.

This module handles downloading and extracting voice recording datasets
from online repositories.
"""

import os
import requests
import tarfile
from pathlib import Path
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceDataDownloader:
    """Handles downloading and extraction of voice recording datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the downloader with a data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_sample_data(self) -> bool:
        """
        Download sample voice recording data.
        
        Note: This is a placeholder implementation as the actual dataset
        requires authentication. In a real scenario, this would implement
        web scraping techniques as suggested in the problem statement.
        
        Returns:
            bool: True if download was successful, False otherwise
        """
        logger.info("Downloading sample voice recording data...")
        
        # Create sample data structure for demonstration
        sample_dir = self.data_dir / "sample_data"
        sample_dir.mkdir(exist_ok=True)
        
        # Create directories for male and female samples
        (sample_dir / "male").mkdir(exist_ok=True)
        (sample_dir / "female").mkdir(exist_ok=True)
        
        logger.info(f"Sample data structure created at {sample_dir}")
        return True
    
    def extract_tgz_files(self, file_path: str, extract_to: Optional[str] = None) -> bool:
        """
        Extract .tgz files to specified directory.
        
        Args:
            file_path: Path to the .tgz file
            extract_to: Directory to extract to (defaults to data_dir)
            
        Returns:
            bool: True if extraction was successful, False otherwise
        """
        if extract_to is None:
            extract_to = self.data_dir
            
        try:
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=extract_to)
            logger.info(f"Successfully extracted {file_path} to {extract_to}")
            return True
        except Exception as e:
            logger.error(f"Failed to extract {file_path}: {e}")
            return False
    
    def list_wav_files(self, directory: str) -> List[str]:
        """
        List all .wav files in a directory recursively.
        
        Args:
            directory: Directory to search for .wav files
            
        Returns:
            List[str]: List of .wav file paths
        """
        wav_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        return wav_files
    
    def get_metadata_files(self, directory: str) -> List[str]:
        """
        Get metadata files (PROMPTS, README, etc.) from the dataset.
        
        Args:
            directory: Directory to search for metadata files
            
        Returns:
            List[str]: List of metadata file paths
        """
        metadata_files = []
        target_files = ['PROMPTS', 'prompts-original', 'README', 'LICENSE']
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file in target_files:
                    metadata_files.append(os.path.join(root, file))
        return metadata_files