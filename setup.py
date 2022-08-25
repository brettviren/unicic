from setuptools import setup, find_packages
from glob import glob
from os.path import basename, splitext
setup(
    name = "unicic",
    version = "0.0.0",
    author = "Brett Viren",
    author_email = "bv@bnl.gov",
    maintainer = "Brett Viren",
    maintainer_email = "bv@bnl.gov",
    description = "Feldman-Cousins Unified Confidence Interval Construction",
    license="AGPL",
    packages=find_packages("src"),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    setup_requires=["pytest","pytest-runner"],
    python_requires='>=3.9',
    install_requires=["numpy","cupy-cuda11x","jax"],
    # long_description = "",
)
