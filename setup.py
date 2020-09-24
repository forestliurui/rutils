from setuptools import setup, find_packages

setup(
  name='RL Utils',
  version='0.1dev',
  packages=find_packages(),
  install_requires=[],
  include_package_data = True,
  package_data = {'': ['logging.conf']},
)