import os
import sys
from setuptools import setup, find_packages

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path)

long_description = None
with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    long_description = f.read()

setup(
    name='quanzi',
    version='0.1',
    description='How to compute guanxi circle',
    long_description=long_description,
    author='yuezhang18',
    author_email='yuezhang18@mails.tsinghua.edu.cn',
    license='MIT',
    packages=find_packages(exclude=('tests', 'tests.*')),
    include_package_data=True,
    classifiers=[
        'Development Status :: 1 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Clustering'
    ],
    install_requires=[
        'pandas',
        'numpy',
        'networkx',
    ]
)
