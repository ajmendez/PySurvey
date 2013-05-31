from distutils.core import setup

setup(
    name='pySurvey',
    version='0.1',
    author='Alexander Mendez',
    author_email='ajmendez@ucsd.edu',
    packages=['pySurvey', ],
    # scripts=['bin/', ],
    url='http://github.com/ajmendez/pySurvey',
    license='LICENSE.txt',
    description='Useful astronomical redshift survey programs and utilities',
    long_description=open('README.md').read(),
    install_requires=[
        "mangle",
        "caldav == 0.1.4",
    ],
)