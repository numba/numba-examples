from setuptools import setup

setup(
    name='numba_bench',
    version='0.1',
    description='A benchmark runner developed for the Numba project',
    packages=['numba_bench'],
    scripts=['bin/numba_bench'],
    include_package_data=True,
)
