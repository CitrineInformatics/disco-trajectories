import os
from setuptools import setup
from setuptools import find_packages


with open(
    os.path.join(
        os.path.dirname(__file__), "src", "discworld", "VERSION.txt"
    )
) as fr:
    version = fr.read().strip()


setup(
    name="discworld",
    version=version,
    description="Tool to simulate discovery trajectories using particle swarm optimization",
    url="https://github.com/CitrineInformatics/disco-trajectories",
    author="Vinay Hegde",
    author_email="vhegde@citrine.io",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["pandas"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
)
