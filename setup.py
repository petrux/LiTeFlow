"""Setup module for the liteflow package."""

from setuptools import setup

setup(name='liteflow',
      version='0.1',
      description='Liteweight TensorFlow extensio library',
      url='https://github.com/petrux/LiTeFlow',
      author='Giulio Petrucci (petrux)',
      author_email='giulio.petrucci@gmail.com',
      license='Apache License 2.0',
      packages=['liteflow'],
      install_requires=[
          'tensorflow'
      ],
      zip_safe=False)
