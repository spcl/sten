# STen: An Interface for Efficient Sparsity in PyTorch

STen aims to solve the following questions that remained unanswered in the current implementation ([torch.sparse](https://pytorch.org/docs/1.11/sparse.html)) of sparsity in PyTorch 1.11.

* How to incorporate a sparsifying strategy in the model and use it in runtime?​
* How to keep sparsity level the same during training?​
* How to enable full autograd support​?
* How to enable custom sparse formats and operator implementations?

## Code organization

The core implementation is located in [sten.py](sten.py). It is tested by running the [pytest](https://docs.pytest.org/) over [test_api.py](test_api.py).

## Examples

* Check [build_from_scratch.ipynb](build_from_scratch.ipynb) to see the example use of interface to build PyTorch module from scratch.
* Check [modify_existing.ipynb](modify_existing.ipynb) to see the example of converting existing dense PyTorch module to sparse.

