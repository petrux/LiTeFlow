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
      install_requires=[
          'appdirs==1.4.3',
          'funcsigs==1.0.2',
          'mock==2.0.0',
          'numpy==1.12.1',
          'packaging==16.8',
          'pbr==2.1.0',
          'protobuf==3.2.0',
          'pyparsing==2.2.0',
          'six==1.10.0',
          'tensorflow==1.1.0',
      ],
      zip_safe=False)
