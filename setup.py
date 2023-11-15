# coding=utf-8
from setuptools import setup, find_packages
import versioneer

setup(name="tiamat",
      version=versioneer.get_version(),
      packages=find_packages(),
      author="Christian Schiffer",
      author_email="c.schiffer@fz-juelich.de",
      url="https://jugit.fz-juelich.de/c.schiffer/tiamat",
      description="tiamat",
      install_requires=[
          "click",
      ],
      entry_points={
          "console_scripts": [
              "tiamat = tiamat.cli:cli",
          ]
      },
      cmdclass=versioneer.get_cmdclass(),
      )
