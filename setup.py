"""Setup module for the liteflow package."""

from setuptools import setup, find_packages

setup(name='liteflow',
      version='0.0.3',
      description='Liteweight TensorFlow extension library',
      url='https://github.com/petrux/LiTeFlow',
      author='Giulio Petrucci (petrux)',
      author_email='giulio.petrucci@gmail.com',
      license='Apache License 2.0',
      packages=find_packages(exclude=["liteflow.tests"]),
      install_requires=[],
      zip_safe=False)
