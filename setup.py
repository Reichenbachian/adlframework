#!/usr/bin/env python

from distutils.core import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session='hack')
reqs = [str(ir.req) for ir in install_reqs]

setup(name='adlframework',
      version='1.0',
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
              "adlframework.controllers.midis",
					],
      install_requires=reqs,
     )
