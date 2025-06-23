from setuptools import setup, find_packages

setup(
    name="scitrust",
    version="2.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'scitrust-run=src.main:main',
            'scitrust-eval=src.evaluation:main'
        ],
    },
)
