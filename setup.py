"""
Setup and build script for C++ accelerated extensions (news_fast_ops).
Compiles csrc/fast_collator.cpp and csrc/fast_metrics.cpp using PyTorch C++ extension.
"""

import os
import sys
from setuptools import setup, find_packages

# Determine compilation flags based on OS
extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/openmp", "/std:c++17"]
else:
    extra_compile_args = ["-O3", "-fopenmp", "-std=c++17"]
    extra_link_args = ["-fopenmp"]

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    ext_modules = [
        CppExtension(
            name="news_fast_ops",
            sources=[
                "csrc/fast_collator.cpp",
                "csrc/fast_metrics.cpp",
                "csrc/bindings.cpp",
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
except ImportError:
    # PyTorch is not yet installed; allow basic setup without extensions
    ext_modules = []
    cmdclass = {}

setup(
    name="news_classifier",
    version="1.0.0",
    description="Modular DistilGPT2 News Topic Classification with C++ Acceleration & MLflow",
    packages=find_packages(include=["src", "src.*", "utils", "utils.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
)
