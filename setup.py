#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:29:06 2020

@author: btreece
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="PyNumericBrad",
    version="0.0.1",
    author="Bradley W. Treece",
    author_email="Bradley.W.Treece@gmail.com",
    description="A package to consolidate some home-made numeric scripts.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bradtreece/PyNumericBrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)