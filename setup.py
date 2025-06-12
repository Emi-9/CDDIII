from setuptools import setup, find_packages

setup(
    name="CDDIII",
    version="0.1.0",
    author="Emiliano",
    packages=find_packages(),

    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "statsmodels"
    ]
)
