import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

scripts = [os.path.join("bin", script) for script in os.listdir("bin")]

setuptools.setup(
    name="c3pp",  # Replace with your own username
    version="1.9.6.3",
    author="Trygve Leithe Svalheim",
    author_email="trygvels@astro.uio.no",
    description="A commander3 postprocessing tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trygvels/c3pp",
    packages=setuptools.find_packages(),
    include_package_data=True,
    scripts=scripts,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'healpy',
        'click',
        'numpy',
        'matplotlib',
        'numba',
        'pathlib',
        'tqdm',
        'pandas',
        'seaborn',
        'cmasher',
        'plotly',
        'brokenaxes',
        'camb',
    ],
    python_requires=">=3.6",
)
