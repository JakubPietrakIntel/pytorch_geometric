import argparse
import ast
import os.path as osp
from timeit import default_timer

import tqdm
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.profile import torch_profile
from contextlib import nullcontext


def run(args: argparse.ArgumentParser) -> None:
    for dataset_name in args.datasets:
        print(f"Dataset: {dataset_name}")
        #root = osp.join(args.root, f'{dataset_name}')

        if dataset_name == 'mag':
            transform = T.ToUndirected(merge=True)
            dataset = OGB_MAG(root=args.root, transform=transform)
            train_idx = ('paper', dataset[0]['paper'].train_mask)
            eval_idx = ('paper', None)
            neighbor_sizes = args.hetero_neighbor_sizes
        else:
            transform = T.ToSparseTensor(
            remove_edge_index=False) if args.use_sparse_tensor else None
            dataset = PygNodePropPredDataset(name=f'ogbn-{dataset_name}', root=args.root, transform=transform)
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train']
            eval_idx = None
            neighbor_sizes = args.homo_neighbor_sizes

        data = dataset[0].to(args.device)

        # for num_neighbors in neighbor_sizes:
        #     print(f'Training sampling with {num_neighbors} neighbors')
        #     for batch_size in args.batch_sizes:
        #         train_loader = NeighborLoader(
        #             data,
        #             num_neighbors=num_neighbors,
        #             input_nodes=train_idx,
        #             batch_size=batch_size,
        #             shuffle=True,
        #             num_workers=args.num_workers,
        #         )
        #         runtimes = []
        #         num_iterations = 0
        #         for run in range(args.runs):
        #             start = default_timer()
        #             for batch in tqdm.tqdm(train_loader):
        #                 num_iterations += 1
        #             stop = default_timer()
        #             runtimes.append(round(stop - start, 3))
        #         average_time = round(sum(runtimes) / args.runs, 3)
        #         print(f'batch size={batch_size}, iterations={num_iterations}, '
        #               f'runtimes={runtimes}, average runtime={average_time}')

        print('Evaluation sampling with all neighbors')
        average_times = []
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
            
            with torch_profile() if args.profile else nullcontext(): #as gs
                with subgraph_loader.enable_cpu_affinity() if args.affinity else nullcontext():
                    for _ in range(args.runs):
                        start = default_timer()
                        for batch in tqdm.tqdm(subgraph_loader):
                            num_iterations += 1
                        stop = default_timer()
                        runtimes.append(round(stop - start, 3))
            average_time = round(sum(runtimes) / args.runs, 3)
            average_times.append(average_time)
            print(f'batch size={batch_size}, iterations={num_iterations}, '
                f'runtimes={runtimes}, average runtime={average_time}')
        print(average_times)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('NeighborLoader Sampling Benchmarking')

    add = parser.add_argument
    add('--device', default='cpu')
    add('--datasets', nargs="+", default=['products']) #['arxiv', 'products', 'mag']
    add('--root', default='../../data')
    add('--batch-sizes', default=[8192, 4096, 2048, 1024, 512],
        type=ast.literal_eval)
    add('--eval-batch-sizes', default=[16384, 8192, 4096, 2048, 1024, 512],
        type=ast.literal_eval)
    add('--homo-neighbor_sizes', default=[[10, 5], [15, 10, 5], [20, 15, 10]],
        type=ast.literal_eval)
    add('--hetero-neighbor_sizes', default=[[5], [10], [10, 5]],
        type=ast.literal_eval)
    add(
        '--use-sparse-tensor', action='store_true',
        help='use torch_sparse.SparseTensor as graph storage format')
    add('--num-workers', type=int, default=0)
    add('--runs', type=int, default=3)
    add('--profile', default=False, action='store_true')
    add('--filter', default=False, action='store_true')
    add('--affinity', default=False, action='store_true')
    run(parser.parse_args())
