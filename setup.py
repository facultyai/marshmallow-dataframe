from setuptools import setup, find_packages

setup(
    name="marshmallow-numerical",
    description="Marshmallow Schema generator for pandas and numpy",
    license="MIT",
    url="https://github.com/marshmallow-numerical",
    author="Víctor Zabalza",
    author_email="vzabalza@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    use_scm_version={"version_scheme": "post-release"},
    setup_requires=["setuptools_scm"],
    install_requires=[
        "marshmallow>=3.0.0rc3",
        "pandas",
        "nummpy"
    ],
    python_requires='>=3.6',
)