import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest
from pathlib import Path


notebooks_list = [
    "build_from_scratch.ipynb",
    "custom_implementations.ipynb",
    "modify_existing.ipynb",
]


@pytest.mark.parametrize("notebook", notebooks_list)
def test_jupyter_notebooks(notebook):
    with Path(__file__).with_name(notebook).open("r") as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor()
        ep.preprocess(nb)


if __name__ == "__main__":
    for n in notebooks_list:
        test_jupyter_notebooks(n)
