"""Setup configuration for the FIT_python package."""

from setuptools import setup, find_packages

setup(
    name="FIT_python",
    version="0.1.0",
    description="FIT Otter: Data import and analysis pipeline",
    author="Frederick Kistner",

    # packages argument pointing to the ``src`` directory
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    python_requires=">=3.11",
    install_requires=[
        # core runtime deps
        "pandas>=1.3",
        "xlrd>=1.2",
        "openpyxl>=3.0",
        "numpy>=1.21",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "optuna",
        "catboost",
        "xgboost",
        "lightgbm",
        "scikit-optimize",
        "scipy",
        "umap-learn",
        "pyarrow",
        "PyQt5",
        "tqdm",
        "tqdm-joblib",
        "joblib",
    ],
    extras_require={
        # dev/testing tools
        "dev": [
            "pytest",
            "coverage",
            "flake8",
            "mypy",
            "missingpy",
        ]
    },
)
