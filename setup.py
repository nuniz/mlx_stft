import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlx_stft",
    version="0.1.1",
    author="Asaf Zorea",
    author_email="zoreasaf@gmail.com",
    description="An implementation of STFT and Inverse STFT in mlx",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/nuniz/mlx_stft",
    packages=setuptools.find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*", "tests.*"]
    ),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.6",
    install_requires=["mlx"],
    extras_require={
        "dev": ["matplotlib", "numpy", "soundfile"],
    },
)
