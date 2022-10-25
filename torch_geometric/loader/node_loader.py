from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
import psutil

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.utils import (
    filter_custom_store,
    filter_data,
    filter_hetero_data,
    get_input_nodes,
    get_numa_nodes_cores
)
from torch_geometric.sampler.base import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.typing import InputNodes


class NodeLoader(torch.utils.data.DataLoader):
    r"""A data loader that performs neighbor sampling from node information,
    using a generic :class:`~torch_geometric.sampler.BaseSampler`
    implementation that defines a :meth:`sample_from_nodes` function and is
    supported on the provided input :obj:`data` object.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        node_sampler (torch_geometric.sampler.BaseSampler): The sampler
            implementation to be used with this loader. Note that the
            sampler implementation must be compatible with the input data
            object.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returning data in each worker's subprocess rather than in the
            main process.
            Setting this to :obj:`True` is generally not recommended:
            (1) it may result in too many open file handles,
            (2) it may slown down data loading,
            (3) it requires operating on CPU tensors.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        node_sampler: BaseSampler,
        input_nodes: InputNodes = None,
        transform: Callable = None,
        filter_per_worker: bool = False,
        use_cpu_worker_affinity=False,
        loader_cores=None,
        compute_cores=None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.data = data
        # NOTE sampler is an attribute of 'DataLoader', so we use node_sampler
        # here:
        self.node_sampler = node_sampler

        # Store additional arguments:
        self.input_nodes = input_nodes
        self.transform = transform
        self.filter_per_worker = filter_per_worker

        # Get input type, or None for homogeneous graphs:
        node_type, input_nodes = get_input_nodes(self.data, input_nodes)
        self.input_type = node_type
        
        # CPU Affinitization for loader and compute cores
        worker_init_fn = None
        if use_cpu_worker_affinity:
            if self.device.type == 'cpu':
                
                self.num_workers = kwargs.get('num_workers', 0)
                loader_cores = loader_cores[:] if loader_cores else []
                compute_cores = compute_cores[:] if compute_cores else []
                
                if not self.num_workers > 0:
                    raise Exception('ERROR: affinity should be used with at least one DataLoader worker')
                if self.loader_cores and len(self.loader_cores) != self.num_workers:
                    raise Exception('ERROR: cpu_affinity incorrect '
                                    'number of loader_cores={} for num_workers={}'
                                    .format(self.loader_cores, self.num_workers))
                
                if not self.loader_cores or not compute_cores:
                    numa_info = get_numa_nodes_cores()
                    if numa_info and len(numa_info[0]) > self.num_workers:
                        # take one thread per each node 0 core
                        node0_cores = [cpus[0] for core_id, cpus in numa_info[0]]
                    else:
                        node0_cores = list(range(psutil.cpu_count(logical = False))) 
                        # TODO: test if using logical cores is feasible

                if len(node0_cores) <= self.num_workers:
                    raise Exception('ERROR: more workers than available cores')

                self.loader_cores = loader_cores or node0_cores[0:self.num_workers]
                compute_cores = [cpu for cpu in node0_cores if cpu not in self.loader_cores]
                
                try:
                    # set compute cores affinity
                    psutil.Process().cpu_affinity(compute_cores)
                    torch.set_num_threads(len(compute_cores))
                    # set worker cores affinity
                    worker_init_fn = self.worker_init_function
            else:
                raise Exception("This function can only be used with CPU device")                
                
        super().__init__(input_nodes, collate_fn=self.collate_fn,
                worker_init_fn=worker_init_fn, **kwargs)

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting (Data or HeteroData) object to be used
        downstream."""
        if isinstance(out, SamplerOutput):
            data = filter_data(self.data, out.node, out.row, out.col, out.edge,
                               self.node_sampler.edge_permutation)
            data.batch = out.batch
            data.batch_size = out.metadata

        elif isinstance(out, HeteroSamplerOutput):
            if isinstance(self.data, HeteroData):
                data = filter_hetero_data(self.data, out.node, out.row,
                                          out.col, out.edge,
                                          self.node_sampler.edge_permutation)
            else:  # Tuple[FeatureStore, GraphStore]
                data = filter_custom_store(*self.data, out.node, out.row,
                                           out.col, out.edge)

            for key, batch in (out.batch or {}).items():
                data[key].batch = batch
            data[self.input_type].batch_size = out.metadata

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(out)}'")

        return data if self.transform is None else self.transform(data)

    def collate_fn(self, index: NodeSamplerInput) -> Any:
        r"""Samples a subgraph from a batch of input nodes."""
        if isinstance(index, (list, tuple)):
            index = torch.tensor(index)

        out = self.node_sampler.sample_from_nodes(index)
        if self.filter_per_worker:
            # We execute `filter_fn` in the worker process.
            out = self.filter_fn(out)
        return out

    def worker_init_function(self, worker_id):
        """Worker init default function.
                Parameters
                ----------
                worker_id : int
                    Worker ID.
                self.loader_cores : [int] (optional)
                    List of cpu cores to which dataloader workers should affinitize to.
                    default: node0_cores[0:num_workers]
        """
        try:
            psutil.Process().cpu_affinity([self.loader_cores[worker_id]])
            #print('CPU-affinity worker {} has been assigned to core={}'
            #        .format(worker_id, self.loader_cores[worker_id]))
        except:
            raise Exception('ERROR: cannot use affinity id={} cpu_cores={}'
                            .format(worker_id, self.loader_cores))
    
    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()
        # We execute `filter_fn` in the main process.
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

          
    def enable_cpu_affinity(self, loader_cores=None, compute_cores=None, verbose=True):
        """ Helper method for enabling cpu affinity for compute threads and dataloader workers
        Only for CPU devices
        Uses only NUMA node 0 by default for multi-node systems
        Parameters
        ----------
        loader_cores : [int] (optional)
            List of cpu cores to which dataloader workers should affinitize to.
            default: node0_cores[0:num_workers]
        compute_cores : [int] (optional)
            List of cpu cores to which compute threads should affinitize to
            default: node0_cores[num_workers:]
        verbose : bool (optional)
            If True, affinity information will be printed to the console
        Usage
        -----
        with dataloader.enable_cpu_affinity():
            <training loop>
        """
        if self.device.type == 'cpu':



           

            try:
                # set compute cores affinity
                psutil.Process().cpu_affinity(compute_cores)
                torch.set_num_threads(len(compute_cores))
                # set worker cores affinity
                worker_init_fn = worker_init_function
                
                self.cpu_affinity_enabled = True
                if verbose:
                    print('{} DL workers are assigned to cpus {}, main process will use cpus {}'
                        .format(self.num_workers, loader_cores, compute_cores))

                yield
            finally:
                # restore omp_num_threads and cpu affinity
                psutil.Process().cpu_affinity(affinity_old)
                set_num_threads(nthreads_old)
                self.worker_init_fn = worker_init_fn_old

                self.cpu_affinity_enabled = False
        else:
            raise Exception("This function can only be used with CPU device")
