from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    entry = project_root / "run_pair_workstation.py"
    excludes = [
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "torch",
        "matplotlib",
        "pytest",
        "IPython",
        "jupyter_client",
        "jupyter_core",
        "nbformat",
        "notebook", "jupyterlab", "dask", "distributed", "bokeh", "plotly",
        "sklearn", "nltk", "imageio", "PIL", "xarray", "sympy", "sqlalchemy",
        "tables", "h5py", "openpyxl", "lxml", "numba", "tensorflow",
        "statsmodels.api", "statsmodels.tests",
        "sphinx", "docutils", "pyarrow", "fsspec", "s3fs", "botocore", "boto3",
        "pandas.io.formats.style", "pandas.io.clipboard", "ipywidgets",
    ]
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name",
        "DualStrategyDesk",
        "--paths",
        str(project_root),
        "--add-data",
        str(project_root / "futures_strategy" / "web" / "pairs") + ";futures_strategy/web/pairs",
    ]
    for module_name in excludes:
        command.extend(["--exclude-module", module_name])
    command.extend(
        [
        str(entry),
        ]
    )
    subprocess.run(command, cwd=project_root, check=True)
    exe_path = project_root / "dist" / "DualStrategyDesk.exe"
    print(f"exe: {exe_path}")


if __name__ == "__main__":
    main()
