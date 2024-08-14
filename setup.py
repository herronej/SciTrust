from setuptools import setup, find_packages

setup(
    name="scitrust",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'scitrust-run=src.main:main',
        ],
    },
)
