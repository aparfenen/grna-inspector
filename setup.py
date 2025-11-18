"""
Setup script for gRNA Inspector package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_file = Path(__file__).parent / "requirements_ml.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

# Read README
readme_file = Path(__file__).parent / "README_ML.md"
if readme_file.exists():
    with open(readme_file) as f:
        long_description = f.read()
else:
    long_description = "gRNA classification using machine learning"

setup(
    name="grna_inspector",
    version="0.1.0",
    description="Machine learning tools for gRNA classification in Trypanosoma brucei",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/aparfenen/grna-inspector",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Dependencies
    install_requires=requirements,
    
    # Python version
    python_requires=">=3.8",
    
    # Entry points (CLI scripts)
    entry_points={
        'console_scripts': [
            'grna-train=grna_inspector.cli:train',
            'grna-predict=grna_inspector.cli:predict',
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    
    # Additional data
    include_package_data=True,
    zip_safe=False,
)
