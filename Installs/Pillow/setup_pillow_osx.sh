#!/bin/bash

# As good practice always update brew first.
brew update 

# Install Pillow's dependencies
pip install --upgrade pip
brew install libjpeg
brew install libtiff
brew install webp
brew install little-cms2

# Install Pillow
pip install Pillow --upgrade
pip install enum34 --upgrade
pip install docutils --upgrade

# For more information, on install Pillow, see below:
# https://pillow.readthedocs.org/en/latest/installation.html
