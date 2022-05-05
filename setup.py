from setuptools.extension import Extension
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy

setup (
    name='gridflag',
    version='0.10',
    packages=find_packages(),
    include_package_data=True,
    include_dirs=[numpy.get_include()],
    entry_points="""
      [console_scripts]
      gridflag=gridflag.gridflag:main
   """,
)
