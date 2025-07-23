from setuptools import setup, find_packages

setup(
    name='snnpy',
    version='0.1.0',
    author='Ruggero Freddi',
    author_email='info@ruggerofreddi.it',
    description='Simulation and analysis of Spiking Neural Networks (SNNs)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RuggeroFreddi/snnpy',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22,<1.27",
        "scipy>=1.8,<1.13",
        "networkx>=2.6,<3.3",
        "pandas>=1.3,<2.3",
        "scikit-learn>=1.1,<1.5",
        "matplotlib>=3.5,<3.9",
        "seaborn>=0.11,<0.13",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
