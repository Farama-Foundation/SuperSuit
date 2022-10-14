"""Setups up the SuperSuit module."""

from setuptools import find_packages, setup


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return header_count, long_description


def get_version():
    """Gets the pettingzoo version."""
    path = "supersuit/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")

version = get_version()
header_count, long_description = get_description()

setup(
    name="SuperSuit",
    version=version,
    author="Farama Foundation",
    author_email="contact@farama.org",
    description="Wrappers for Gymnasium and PettingZoo",
    url="https://github.com/Farama-Foundation/SuperSuit",
    license_files=("LICENSE.txt",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Reinforcement Learning", "game", "RL", "AI", "gymnasium"],
    python_requires=">=3.7, <3.11",
    packages=["pettingzoo"]
    + ["pettingzoo." + pkg for pkg in find_packages("pettingzoo")],
    include_package_data=True,
    install_requires=["numpy>=1.18.0", "gymnasium>=0.26.0"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    extras_require=extras,

    name="SuperSuit",
    version=get_version(),
    author="SuperSuit Community",
    author_email="jkterry@farama.org",
    description="Wrappers for Gymnasium and PettingZoo",
    license_files=("LICENSE.txt",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PettingZoo-Team/SuperSuit",
    keywords=["Reinforcement Learning", "gymnasium"],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "pettingzoo>=1.22.0",
        "tinyscaler>=1.0.4",
        "gymnasium>=0.26.0",
        "pygame==2.1.2",
        "pymunk==6.2.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras={"dev": ["pettingzoo[butterfly]"]},
    include_package_data=True,
)
