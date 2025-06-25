from setuptools import find_packages, setup

setup(
    name="pylint_nan_check",
    version="0.1",
    packages=find_packages(),
    # If pylint_nan_check is not a real package, do not include it in install_requires
    # install_requires=[
    #     ...
    #     # 'pylint_nan_check',
    #     ...
    # ]
    install_requires=[
        "pylint>=2.0.0",
        "astroid>=2.0.0",
    ],
)
