# IDAM module for Python
# B.Dudson, University of York, July 2009
# Released under the BSD license

# Set the path to the IDAM library here
idamdir = '/home/ben/codes/idam'

#########################################

from distutils.core import setup, Extension

module1 = Extension('idam',
                    include_dirs = [idamdir],
                    library_dirs = [idamdir],
                    libraries = ['idam'],
                    sources = ['idammodule.c'])

setup (name = 'IDAM',
       version = '1.0',
       description = 'Package to read IDAM data',
       long_description = "Provides an interface to D.G.Muir's IDAM library for reading MAST data",
       author = "B.Dudson, University of York",
       author_email = "bd512@york.ac.uk",
       url = "http://www-users.york.ac.uk/~bd512/",
       classifiers = [
           'Environment :: Console',
           'Intended Audience :: End Users/Desktop',
           'Intended Audience :: Developers',
           'License :: OSI Approved :: BSD License',
           'Operating System :: POSIX',
           'Programming Language :: Python',
           ],
       data_files=[('',[])],
       ext_modules = [module1])
