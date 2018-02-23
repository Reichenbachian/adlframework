#!/usr/bin/env python

from setuptools import setup
#from distutils.core import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session='hack')
reqs = [str(ir.req) for ir in install_reqs]

setup(name='adlframework',
      version='1.4',
      description='Deep learning Streamlined Process',
      author='George Alexander Reichenbach',
      author_email='greichenbach@andover.edu',
      url='https://student.andover.edu',
      packages=['adlframework',
    	        "adlframework.retrievals",
              "adlframework.augmentations",
              "adlframework.dataentity",
              "adlframework.nets",
              "adlframework.caches",
              "adlframework.controllers",
              "adlframework.controllers.audio",
              "adlframework.controllers.arff",
              "adlframework.controllers.classification",
              "adlframework.controllers.fits",
              "adlframework.controllers.general",
              "adlframework.controllers.images",
              "adlframework.controllers.lstm",
              "adlframework.controllers.midis"],
      install_requires=reqs,
      keywords = ['deep learning', 'data framework', 'data', 'data manipulation', 'data augmentation'],
      download_url = 'https://github.com/Reichenbachian/adlframework/archive/0.1.tar.gz' 
)
