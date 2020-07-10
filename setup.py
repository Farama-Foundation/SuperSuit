import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SuperSuit",
    version="0.1.13",
    author="PettingZoo Team",
    author_email="justinkterry@gmail.com",
    description="Wrappers for Gym and PettingZoo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PettingZoo-Team/SuperSuit",
    keywords=["Reinforcement Learning", "gym"],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=["pettingzoo>=0.1.9", "scikit-image"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
