from setuptools import setup, find_packages
import unittest
import codecs
import re
import os


here = os.path.abspath(os.path.dirname(__file__))


def test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        os.path.join(here, "tests"),
        pattern="test_*.py"
    )

    return test_suite


def find_version():
    """Find version in teapot/__init__.py"""
    with codecs.open(os.path.join(here, "teapot", "__init__.py"), 'r') as fp:
        version_file = fp.read()
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]",
            version_file, re.M
        )
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="teapot-nlp",
    version=find_version(),
    description="Source and target side evaluation of adversarial attacks on "
    "NLP models",
    long_description=codecs.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pmichel31415/teapot-nlp",
    author="Paul Michel",
    license="MIT",
    test_suite="setup.test_suite",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Text Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "teapot=teapot.main:main",
        ],
    },
    install_requires=[
        "sacrebleu>=1.3.1",
    ],
    include_package_data=True,
)
