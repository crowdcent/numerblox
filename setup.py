from setuptools import setup
from setuptools import find_packages


def load(path):
    return open(path, 'r').read()


classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering"]


if __name__ == "__main__":
    setup(
        name="numerblox",
        version="1.0",
        maintainer="crowdcent",
        maintainer_email="support@crowdcent.com",
        description="Solid Numerai Pipelines",
        long_description=load('README.md'),
        long_description_content_type='text/markdown',
        url='https://github.com/crowdcent/numerblox',
        platforms="OS Independent",
        classifiers=classifiers,
        license='MIT License',
        package_data={'numerai': ['LICENSE', 'README.md']},
        packages=find_packages(exclude=['tests']),
        # TODO Set requirements
        install_requires=[],
        )
