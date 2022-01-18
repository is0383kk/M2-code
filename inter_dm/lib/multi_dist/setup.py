from distutils.core import setup, Extension
module1 = Extension('multi_dist',
                    sources = ['multiWrapper.c'])

setup(name = 'multi_dist',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])

       