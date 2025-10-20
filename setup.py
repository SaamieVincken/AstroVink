from setuptools import setup, find_packages

setup(
    name="astro_vink",
    version="1.0.0",
    author="Saamie Helena Vincken",
    author_email="saamie.vincken@fhnw.ch",
    description="AstroVink: A vision transformer approach to find strong gravitational lens systems.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/saamievincken/AstroVink-Q1",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "pillow>=10.0",
        "scikit-learn>=1.3",
        "tqdm>=4.65",
        "matplotlib>=3.7"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
)
