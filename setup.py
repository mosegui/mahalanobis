import os
import setuptools

_readme_path = os.path.join(os.path.dirname(__file__), "README.md")

with open(_readme_path) as readme:
    long_description = readme.read()

setuptools.setup(
     name='mahalanobis',
     version='1.2.0',
     packages=setuptools.find_packages(),
     author="Daniel Moseguí González",
     author_email="d.mosegui@gmail.com",
     description="Package for performing calculations of Mahalanobis distances",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/mosegui/mahalanobis",
     classifiers=[
	 "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    python_requires='>=3.6',
 )
