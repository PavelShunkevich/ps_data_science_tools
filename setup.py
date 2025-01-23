from setuptools import setup, find_packages

def readme():
  with open('README.md', 'r') as f:
    return f.read()

setup(
    name="ps_data_science_tools",
    version="0.1.1",
    author="Pavel Shunkevich",
    author_email="pavel.shunkevich@gmail.com",
    description="A short description of my Data Science library wil be in the future",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/PavelShunkevich/ps_data_science_tools.git",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
    ],
    python_requires='>=3.12'
)