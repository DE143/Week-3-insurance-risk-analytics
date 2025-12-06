from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="insurance-risk-analytics",
    version="1.0.0",
    author="Insurance Analytics Team",
    author_email="analytics@alphacare.co.za",
    description="Insurance risk analytics and predictive modeling system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/insurance-risk-analytics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "insurance-pipeline=scripts.run_pipeline:main",
        ],
    },
    include_package_data=True,
)