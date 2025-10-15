from setuptools import setup, find_packages

setup(
    name="astro_vink",
    version="1.0.0",
    author="Saamie Helena Vincken",
    description="AstroVink: A vision transformer approach to find strong gravitational lens systems.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "scikit-learn",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
