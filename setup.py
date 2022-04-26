import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PathwayAnalysis",
    version="0.0.3",
    author="Nate Mankovich",
    author_email="Nate.Mankovich@colostate.edu",
    description="A package for biological pathway analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nmank/PathwayAnalysis",
    project_urls={
        "Bug Tracker": "https://github.com/nmank/PathwayAnalysis/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "./"},
    packages=setuptools.find_packages("./"),
    python_requires=">=3.7",
)
