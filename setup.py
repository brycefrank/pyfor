from setuptools import setup, find_packages, Extension

setup(
    name="pyfor",
    version="0.3.5",
    author="Bryce Frank",
    author_email="bfrank70@gmail.com",
    packages=["pyfor", "pyfortest"],
    url="https://github.com/brycefrank/pyfor",
    license="LICENSE.txt",
    description="Tools for forest resource point cloud analysis.",
    install_requires=["laspy", "laxpy", "python-coveralls"],  # Dependencies from pip
)
