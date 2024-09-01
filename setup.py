from setuptools import setup, find_packages

setup(
    name='nsoran',
    version='0.1',
    packages=find_packages(exclude=['tests']),
    description='Gymnasium environment for ns-o-ran',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Andrea Lacava, Tommaso Pietrosanti',
    author_email='lacava.a@northeastern.edu',
    url='https://github.com/wineslab/ns-o-ran-gym',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
