import setuptools

with open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setuptools.setup(
    name="cleo",
    version="0.0.1",
    author="Sebastian Wehrle",
    author_email="sebastian.wehrle@boku.ac.at",
    descripton="Cleo supports wind resource assessment with the Global Wind Atlas",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/sebwehrle/cleo",
    project_urls={},
    license="MIT License",
    packages=["cleo"],
    package_data={'cleo': ['resources/*.yml']},
    install_requires=[
        "urllib3",
        "certifi",
        "tqdm",
        "netcdf4",
        "scipy",
        "numpy",
        "rasterio",
        "pandas",
        "openpyxl",
        "xarray",
        "rioxarray",
        "geopandas",
        "pycountry",
        "pint",
    ]
)
