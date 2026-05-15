from distutils.core import setup

def read_requirements():
    """Read the requirements.txt file and return a list of dependencies."""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return fh.read().splitlines()

setup(name='AdaptMol',
      version='1.0.0',
      description='AdaptMol',
      author='Feng Hu',
      url='https://github.com/fffh1/AdaptMol',
      packages=['adaptmol'],
      package_dir={'adaptmol': 'adaptmol'},
      package_data={'adaptmol': ['vocab/*']},
      python_requires='>=3.8',
      setup_requires=['numpy'],
      install_requires=read_requirements(),
      )
