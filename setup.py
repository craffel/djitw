from setuptools import setup

setup(
    name='djitw',
    version='0.0.0',
    description='Fast just-in-time compiled DTW',
    author='Colin Raffel',
    author_email='craffel@gmail.com',
    url='https://github.com/craffel/djitw',
    packages=['djitw'],
    long_description="""\
    Fast just-in-time compiled dynamic time warping routines.
    """,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords='dtw',
    license='GPL',
    install_requires=[
        'numpy >= 1.7.0',
        'numba >= 0.18.2'
    ],
)
