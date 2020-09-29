import setuptools

with open("README.md", "r") as fh:
        long_description = fh.read()

        setuptools.setup(
                name="minimizer",
                version="0.0.1",
                author="Fergus Barratt",
                description="minimization algorithms",
                long_description=long_description,
                long_description_content_type="text/markdown",
                url="https://github.com/fergusbarratt/minimizer",
                packages=setuptools.find_packages(),
                classifiers=[
                            "Programming Language :: Python :: 3",
                            "License :: OSI Approved :: GNU GPLv3",
                            "Operating System :: OS Independent",
                        ],
                python_requires='>=3.6',
        )
