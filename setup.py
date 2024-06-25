from distutils.core import setup
from setuptools import find_packages

setup(
    name='blueforestback',
    version='0.1',
    packages=find_packages(),
    license='Copyright (c) 2023 Hieu Pham',
    zip_safe=True,
    description='Blueforest backend.',
    long_description='Blueforest backend.',
    long_description_content_type='text/markdown',
    author='Hieu Pham',
    author_email='64821726+hieupth@users.noreply.github.com',
    url='https://gitlab.com/hieupth/blueforestback',
    keywords=[],
    install_requires=[],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3'
    ],
)