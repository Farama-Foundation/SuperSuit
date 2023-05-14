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
    python_requires=">=3.7, <3.12",
    packages=find_packages(),
    install_requires=["numpy>=1.19.0", "gymnasium>=0.26.0", "tinyscaler>=1.2.5"],
    extras={
        "dev": [
            "pettingzoo[butterfly] @ git+https://github.com/Farama-Foundation/PettingZoo.git"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
