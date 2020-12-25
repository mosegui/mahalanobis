import setuptools

setuptools.setup(
     name='mahalanobis',
     version='1.0.1',
     packages=setuptools.find_packages(),
     author="Daniel Moseguí González",
     author_email="d.mosegui@gmail.com",
     description="Package for performing calculations of Mahalanobis distances",
     url="https://github.com/mosegui/mahalanobis",
     classifiers=[
	 "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
    python_requires='>=3.6',
 )
