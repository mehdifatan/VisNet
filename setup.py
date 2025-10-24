"""
Setup script for VisNet library.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="visnet",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A biologically-inspired hierarchical vision network library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/visnet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
)

