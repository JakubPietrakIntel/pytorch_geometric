import argparse
import ast
import os.path as osp
from contextlib import nullcontext

from timeit import default_timer

import tqdm
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.profile import torch_profile


def run(args: argparse.ArgumentParser) -> None:
    for dataset_name in args.datasets:
        print(f"Dataset: {dataset_name}")
        root = osp.join(args.root, dataset_name)

        if dataset_name == 'mag':
            transform = T.ToUndirected(merge=True)
            dataset = OGB_MAG(root=root, transform=transform)
            train_idx = ('paper', dataset[0]['paper'].train_mask)
            eval_idx = ('paper', None)
            neighbor_sizes = args.hetero_neighbor_sizes
        else:
            dataset = PygNodePropPredDataset(f'ogbn-{dataset_name}', root)
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train']
            eval_idx = None
            neighbor_sizes = args.homo_neighbor_sizes

        data = dataset[0].to(args.device)
        if neighbor_sizes:
            for num_neighbors in neighbor_sizes:
                print(f'Training sampling with {num_neighbors} neighbors')
                for batch_size in args.batch_sizes:
                    train_loader = NeighborLoader(
                        data,
                        num_neighbors=num_neighbors,
                        input_nodes=train_idx,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        filter_per_worker=args.filter
                    )
                    runtimes = []
                    num_iterations = 0
                    with torch_profile() if args.profile else nullcontext():
                        with subgraph_loader.enable_cpu_affinity(
                        loader_cores=args.loader_cores
                        ) if args.affinity else nullcontext():
                            for run in range(args.runs):
                                start = default_timer()
                                for batch in tqdm.tqdm(train_loader):
                                    num_iterations += 1
                                stop = default_timer()
                                runtimes.append(round(stop - start, 3))
                            average_time = round(sum(runtimes) / args.runs, 3)
                            print(f'batch size={batch_size}, iterations={num_iterations}, '
                                f'runtimes={runtimes}, average runtime={average_time}')

        if args.eval_batch_sizes:
            print('Evaluation sampling with all neighbors')
            for batch_size in args.eval_batch_sizes:
                subgraph_loader = NeighborLoader(
                    data,
                    num_neighbors=[-1],
                    input_nodes=eval_idx,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    filter_per_worker=args.filter
                )
                runtimes = []
                num_iterations = 0
                with torch_profile() if args.profile else nullcontext():
                    with subgraph_loader.enable_cpu_affinity(
                        loader_cores=args.loader_cores
                        ) if args.affinity else nullcontext():
                        for run in range(args.runs):
                            start = default_timer()
                            for batch in tqdm.tqdm(subgraph_loader):
                                num_iterations += 1
                            stop = default_timer()
                            runtimes.append(round(stop - start, 3))
                        average_time = round(sum(runtimes) / args.runs, 3)
                        print(f'batch size={batch_size}, iterations={num_iterations}, '
                            f'runtimes={runtimes}, average runtime={average_time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('NeighborLoader Sampling Benchmarking')

    add = parser.add_argument
    add('--device', default='cpu')
    add('--datasets', nargs="+", default=['arxiv', 'products', 'mag'])
    add('--root', default='../data')
    add('--batch-sizes', default=[8192, 4096, 2048, 1024, 512],
        type=ast.literal_eval)
    add('--eval-batch-sizes', default=[16384, 8192, 4096, 2048, 1024, 512],
        type=ast.literal_eval)
    add('--homo-neighbor_sizes', default=[[10, 5], [15, 10, 5], [20, 15, 10]],
        type=ast.literal_eval)
    add('--hetero-neighbor_sizes', default=[[5], [10], [10, 5]],
        type=ast.literal_eval)
    add('--use-sparse-tensor', action='store_true',
        help='use torch_sparse.SparseTensor as graph storage format')
    add('--num-workers', type=int, default=0,
        help="Number of DataLoader workers to use.")
    add('--runs', type=int, default=3, 
        help="Number of iterations for each test setting.")
    add('--profile', default=False, action='store_true',
        help="Run torch.profiler.")
    add('--filter', default=False, action='store_true',
        help="Use filter per worker.")
    add('--affinity', default=False, action='store_true',
        help="Use DataLoader affinitzation.")
    run(parser.parse_args())
