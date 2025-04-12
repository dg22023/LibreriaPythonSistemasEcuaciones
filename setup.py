import readme
from setuptools import setup, find_packages

setup(
    name="mi_calculadora_de_ecuaciones_lineales_&_no_lineales",
    version="0.1.0",
    description="Una librería para resolver sistemas de ecuaciones lineales y no lineales",
    long_description=readme.read(),
    long_description_content_type="text/markdown",
    author="Gabriel Diaz",
    author_email="dg22023@ues.edu.sv",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    python_requires=">=3.6",
)