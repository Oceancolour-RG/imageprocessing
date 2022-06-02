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
    python_requires=">=3.8, <3.10",
    # packages=find_packages(),
    packages=find_packages(exclude=("tests", "tests.*")),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "imageio>=2.19.3",
        "requests>=2.27.1",
        "numpy>=1.22.4",
        "opencv-python>=4.5.5.64",
        "pysolar>=0.10",
        "matplotlib>=3.5.2",
        "scikit-image>=0.19.2",
        "packaging>=21.3",
        "pyexiftool<=0.4.13",  # this will eventually be removed
        "py3exiv2>=0.11.0",  # replace pyexiftool
        "pytz>=2022.1",
        "pyzbar-x>=0.2.1",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "rasterio>=1.2.10",
    ],
)
