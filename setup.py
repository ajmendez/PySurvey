from distutils.core import setup

setup(
    name='pySurvey',
    version='0.1',
    author='Alexander Mendez',
    author_email='blue.space@gmail.com',
    packages=['pySurvey', ],
    url='http://github.com/ajmendez/pySurvey',
    license='LICENSE.txt',
    description='Useful astronomical redshift survey programs and utilities',
    long_description=open('README.md').read(),
    install_requires=[
        'mangle',
        'numpy',
        'matplotlib',
        'pywcs',
        # 'pyfits',
        # 'git+https://github.com/leejjoon/pywcsgrid2.git',
        
    ],
)