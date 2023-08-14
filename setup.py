from setuptools import setup

def readme():
  with open('README.md') as f:
    return f.read()

setup(name='ptltSNEcv',
      version='0.1',
      description='Parametric time-lagged t-SNE using artificial neural networks for development of collective variables of molecular systems',
      long_description=readme(),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
      ],
      keywords='artificial neural networks molecular dynamics simulation',
      url='https://github.com/spiwokv/ptltSNEcv',
      author='Vojtech Spiwok, ',
      author_email='spiwokv@vscht.cz',
      license='GPLv3+',
      packages=['ptltSNEcv'],
      scripts=['bin/ptltSNEcv'],
      install_requires=[
          'numpy',
          'cython',
          'mdtraj',
          'keras',
          'argparse',
          'datetime',
          'codecov',
      ],
      include_package_data=True,
      zip_safe=False)

