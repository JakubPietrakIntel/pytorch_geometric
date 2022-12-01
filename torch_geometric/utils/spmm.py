from typing import Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from .sparse import is_torch_sparse_tensor


@torch.jit._overload
def spmm(src, other, reduce):
    # type: (Tensor, Tensor, str) -> Tensor
    pass


@torch.jit._overload
def spmm(src, other, reduce):
    # type: (SparseTensor, Tensor, str) -> Tensor
    pass


def spmm(
    src: Union[SparseTensor, Tensor],
    other: Tensor,
    reduce: str = "sum",
) -> Tensor:
    """Matrix product of sparse matrix with dense matrix.

    Args:
        src (Tensor or torch_sparse.SparseTensor]): The input sparse matrix,
            either a :class:`torch_sparse.SparseTensor` or a
            :class:`torch.sparse.Tensor`.
        other (Tensor): The input dense matrix.
        reduce (str, optional): The reduce operation to use
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`
    """
    assert reduce in ['sum', 'add', 'mean', 'min', 'max'], f"Uknown reduction type {reduce}. Supported: ['sum','mean','max','min']"
    reduce = 'sum' if reduce == 'add' else reduce
    
    if isinstance(src, SparseTensor):
        src = src.to_torch_sparse_csr_tensor(dtype=other.dtype)
        
    # if not is_torch_sparse_tensor(src):
    #         raise ValueError("`src` must be a `torch.sparse.Tensor`"
    #         f"or a  (got {type(src)}).")
    
    # TODO: Revise type chcks when torch.sparse.Tensor is available
    
    if not src.layout == torch.sparse_csr:
        raise ValueError(f"src must be a `torch.Tensor` with `torch.sparse_csr` layout {src.layout}")
    return torch.sparse.spmm_reduce(src, other, reduce)

    
SparseTensor.spmm = lambda self, other, reduce="sum": spmm(self, other, reduce)
