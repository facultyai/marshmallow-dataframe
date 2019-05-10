from os import path
from setuptools import setup, find_packages

with open(
    path.join(path.abspath(path.dirname(__file__)), "README.md"),
    encoding="utf-8",
) as f:
    long_description = f.read()

setup(
    name="marshmallow-dataframe",
    description="Marshmallow Schema generator for pandas and numpy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/zblz/marshmallow-dataframe",
    author="Víctor Zabalza",
    author_email="vzabalza@gmail.com",
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
