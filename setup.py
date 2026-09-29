"""
Setup and build script for C++ accelerated extensions (news_fast_ops).
Compiles csrc/fast_collator.cpp and csrc/fast_metrics.cpp using PyTorch C++ extension.
"""

import os
import sys
import glob
from setuptools import setup, find_packages

# Helper: Ensure MSVC cl.exe is in PATH on Windows to avoid WinError 2 in PyTorch cpp_extension
if sys.platform == "win32":
    cl_found = any(
        os.path.exists(os.path.join(p, "cl.exe"))
        for p in os.environ.get("PATH", "").split(os.pathsep)
    )
    if not cl_found:
        # Search for typical MSVC install paths
        msvc_patterns = [
            r"C:\Program Files\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\Hostx64\x64",
            r"C:\Program Files (x86)\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\Hostx64\x64",
            r"C:\Program Files\Microsoft Visual Studio\*\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64",
        ]
        for pattern in msvc_patterns:
            matches = sorted(glob.glob(pattern), reverse=True)
            if matches:
                os.environ["PATH"] = matches[0] + os.pathsep + os.environ.get("PATH", "")
                break

# PyTorch 2.x headers with recent MSVC toolsets require C++20 and /bigobj
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/openmp", "/std:c++20", "/bigobj", "/EHsc"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-fopenmp", "-std=c++20", "-fPIC"]
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
