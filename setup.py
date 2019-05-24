from os import path

from setuptools import find_packages, setup

with open(
    path.join(path.abspath(path.dirname(__file__)), "README.md"),
    encoding="utf-8",
) as f:
    long_description = f.read()

setup(
    name="marshmallow-dataframe",
    description="Marshmallow Schema generator for pandas dataframes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache Software License",
    url="https://github.com/facultyai/marshmallow-dataframe",
    author="Faculty",
    author_email="opensource@faculty.ai",
    packages=find_packages("src"),
    package_dir={"": "src"},
    use_scm_version={
        "version_scheme": "post-release",
        "local_scheme": "dirty-tag",
    },
    setup_requires=["setuptools_scm"],
    install_requires=["marshmallow[reco]>=3.0.0rc4", "pandas", "numpy"],
    python_requires=">=3.6",
)
