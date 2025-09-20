from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="voice-recording-analysis",
    version="1.0.0",
    author="Voice Analysis Team",
    description="A comprehensive toolkit for voice recording analysis and gender classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yss107/voice-recording-analysis",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.9.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "audio": [
            "librosa>=0.10.0",
            "soundfile>=0.10.0",
        ],
        "interactive": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "plotly>=5.0.0",
        ],
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "voice-analysis-test=voice_analysis.test:main",
        ],
    },
)