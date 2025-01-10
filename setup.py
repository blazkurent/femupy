from setuptools import setup, find_packages

setup(
    name='femupy',
    version='0.1.1',
    author='Blaž Kurent',
    author_email='bkurent@fgg.uni-lj.si',
    description='Finite element model updating using natural frequencies and mode shapes',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/blazkurent/femupy',
    packages=find_packages(),
    py_modules=['femupy'],
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'matplotlib',
        'tensorflow',
        'scikit-learn',
        'scipy',
        'SALib',
        'pymoo'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
    ],
)
