from setuptools import find_packages, setup

setup(
    name="pylint_nan_check",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'pylint.plugins': [
            'pylint_nan_check = pylint_nan_check.plugin'
        ]
    },
    install_requires=[
        "pylint>=2.0.0",
        "astroid>=2.0.0",
    ],
)
