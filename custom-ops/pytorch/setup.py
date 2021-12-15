# -*- coding: utf-8 -*-
import os
import re
import subprocess
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="PimPytorch",
    version="1.0.0",
    description="Pim Pytorch library",
    long_description="",
    zip_safe=False,

    packages=['pim_pytorch'],
    package_dir={'pim_pytorch':'./'}
)
