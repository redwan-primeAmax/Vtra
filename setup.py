from setuptools import setup, find_packages

setup(
    name="auto_dubber",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "faster-whisper",
        "transformers",
        "demucs",
        "edge-tts",
        "soundfile",
        "numpy",
        "ffmpeg-python"
    ],
    author="Your Name",
    description="Automated Video Dubbing Tool (English to Bengali)",
)
