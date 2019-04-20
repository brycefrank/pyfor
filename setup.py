from setuptools import setup, find_packages, Extension
import pyfor

setup(
    name='pyfor',
    version=pyfor.__version__,
    author='Bryce Frank',
    author_email='bfrank70@gmail.com',
    packages=['pyfor', 'pyfortest'],
    url='https://github.com/brycefrank/pyfor',
    license='LICENSE.txt',
    description='Tools for forest resource point cloud analysis.',
    install_requires = [ # Dependencies from pip
        'laspy',
        'laxpy',
        'python-coveralls',
        ''
    ]
)
