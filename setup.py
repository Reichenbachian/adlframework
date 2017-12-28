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
              "adlframework.dataentity",
              "adlframework.nets",
              "adlframework.processors",
              "adlframework.callbacks",
              "adlframework.filters"
					],
      install_requires=reqs,
     )
