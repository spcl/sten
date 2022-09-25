# STen: An Interface for Efficient Sparsity in PyTorch

STen aims to solve the following questions that remained unanswered in the current implementation ([torch.sparse](https://pytorch.org/docs/1.11/sparse.html)) of sparsity in PyTorch 1.11.

* How to incorporate a sparsifying strategy in the model and use it in runtime?​
* How to keep sparsity level the same during training?​
* How to enable full autograd support​?
* How to enable custom sparse formats and operator implementations?

## Examples

* Check [build_from_scratch.ipynb](examples/build_from_scratch.ipynb) to see the example use of interface to build PyTorch module from scratch.
* Check [modify_existing.ipynb](examples/modify_existing.ipynb) to see the example of converting existing dense PyTorch module to sparse.
* Check [custom_implementations.ipynb](examples/custom_implementations.ipynb) to see the example of registering custom implementations for sparsifiers and operators that match specific formats of input and output tensors.

## Quick start

```
git clone https://github.com/spcl/sten.git
cd sten
python -m venv venv
source venv/bin/activate
pip install .
python tests/test_api.py
```

## Installation

```
pip install sten
```

## Code organization

The core implementation is located in [sten.py](src/sten/sten.py). Jupyter notebook examples are located in [examples](examples) directory. Even more examples can be found in form of tests in [tests](tests) directory. Tests can be run by calling `pytest` in the project root.


