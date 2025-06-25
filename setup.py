from setuptools import setup, find_packages

setup(
    name='black_scholes',
    version='0.1',
    description='Black-Scholes option pricing model in Python',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
)
