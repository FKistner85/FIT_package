from setuptools import setup, find_packages

setup(
    name="FIT_python",
    version="0.1.0",
    description="FIT Otter: Data import and analysis pipeline",
    author="Frederick Kistner",

    # ← Nur EIN packages-Argument!
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    install_requires=[
        "pandas>=1.3",
        "xlrd>=1.2",
        "openpyxl>=3.0",
        "numpy>=1.21",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "optuna",
        "catboost",
        "lightgbm"
    ],
    python_requires='>=3.11',
)
