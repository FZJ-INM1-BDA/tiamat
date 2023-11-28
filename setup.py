# coding=utf-8
from setuptools import setup, find_packages
import versioneer

setup(name="tiamat",
      version=versioneer.get_version(),
      packages=find_packages(),
      author="Christian Schiffer",
      author_email="c.schiffer@fz-juelich.de",
      url="https://jugit.fz-juelich.de/inm-1/bda/tiamat/tiamat.git",
      description="tiamat",
      setup_requires=[
          "numpy>=1.25.0",
          "cython==0.29.32",
      ],
      install_requires=[
          "click",
          "numpy>=1.25.0",
          "opencv-python-headless>=4.8.1",
          "pytiff",
          "matplotlib",
      ],
      entry_points={
          "console_scripts": [
              "tiamat = tiamat.cli:cli",
          ]
      },
      cmdclass=versioneer.get_cmdclass(),
      )
