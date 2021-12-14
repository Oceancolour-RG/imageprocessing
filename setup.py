#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

HERE = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(HERE , "README.md"), "r") as f:
    README = f.read()

# Parse the version from the main __init__.py
with open("micasense/__init__.py") as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue


# Note, current version of:
# * pysolar does not support >=3.9,
# * numpy does not support <=3.6
# * pyzbar does not support >=3.8, hence pyzbar-x used instead
# * pathlib does not support >= 3.5, hence pathlib2 used instead
# * rawpy does not support >=3.8 , hence removed and replaced with opencv
# The versions in install_requires=[...] were taken from
# https://pypi.org/search/ with filtering for python 3.7 and 3.8

setup(
    name="micasense",
    version=version,
    description=u"Micasense Image Processing",
    long_description=README,
    long_description_content_type="text/markdown",
    author=u"MicaSense, Inc. & Rodrigo A. Garcia",
    author_email="rodrigo.garcia@uwa.edu.au",
    url="https://github.com/Oceancolour-RG/imageprocessing",
    license="MIT",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    python_requires=">=3.7, <3.9",
    # packages=find_packages(),
    packages=find_packages(exclude=("tests", "tests.*")),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "six>=1.16.0",  # requirement of pathlib2
        "imageio>=2.13.3",
        "requests>=2.26.0",
        "numpy>=1.21.3",
        "opencv-python>=4.5.4.60",
        "pysolar>=0.10",  # does not support 3.9
        "matplotlib>=3.5.0",
        "scikit-image>=0.19.0",
        "packaging>=21.3",
        "pyexiftool>=0.4.11",  # this will eventually be removed
        "py3exiv2>=0.9.3",  # replace pyexiftool
        "pytz>=2021.3",
        "pyzbar-x>=0.2.1",
        "tqdm>=4.62.3",
        "pathlib2>=2.3.6",
        "pyyaml>=6.0",
        "rasterio>=1.2.10",
    ],
)
