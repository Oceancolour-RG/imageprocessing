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
# * numpy does not support <=3.6
# * pyzbar was switched to pyzbar-x
# * pyexiftool 0.5.3. works upto python 3.9, however
#   imageprocessing requires pyexiftool<=0.4.13...
# The versions in install_requires=[...] were taken from
# https://pypi.org/search/ with filtering for python 3.8 and 3.9

setup(
    name="micasense",
    version=version,
    description=u"Micasense Image Processing",
    long_description=README,
    long_description_content_type="text/markdown",
    author=u"MicaSense, Inc.",
    author_email="github@micasense.com",
    url="https://github.com/Oceancolour-RG/imageprocessing",
    license="MIT",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    python_requires=">=3.8, <=3.12",
    # packages=find_packages(),
    packages=find_packages(exclude=("tests", "tests.*")),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "imageio>=2.27.0",
        "requests>=2.28.0",
        "numpy>=1.24.0",
        "opencv-python-headless>=4.7.0.72",
        "pysolar>=0.11",
        "matplotlib>=3.7.0",
        "scikit-image>=0.20.0",
        "packaging>=23.0",
        "pyexiftool>=0.5.5",  # this will eventually be removed
        "py3exiv2>=0.11.0",  # replace pyexiftool
        "pytz>=2023.3",
        "pyzbar-x>=0.2.1",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "rasterio>=1.3.6",
    ],
)
