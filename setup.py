from setuptools import find_packages, setup

setup(
    name="pylint_nan_check",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pylint>=2.0.0",
        "astroid>=2.0.0",
    ],
)
