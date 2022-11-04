import argparse
import torch
#print(torch.__config__.parallel_info())
from utils import get_dataset, get_model

from torch_geometric.loader import NeighborLoader
from torch_geometric.profile import timeit

supported_sets = {
    'rgcn':'ogbn-mag',
    'gat':'Reddit',
    'gcn':'Reddit'
}

def run(args: argparse.ArgumentParser) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CPU affinity     
    use_cpu_affinity = (True if args.cpu_affinity == 1 else False)
    # Sparse tensor
    use_sparse_tensor = (True if args.use_sparse_tensor == 1 else False)
    
    print('BENCHMARK STARTS')
   
    for model_name in args.models:
        
        print(f'Evaluation bench for {model_name}:')
        if not args.datasets:
            dataset_name = supported_sets.get(model_name, None)
        else: 
            dataset_name = args.datasets[0]
        
        dataset, num_classes = get_dataset(dataset_name, args.root,
                                        use_sparse_tensor, args.bf16)
        data = dataset.to(device)
        hetero = True if dataset_name == 'ogbn-mag' else False
        mask = ('paper', None) if dataset_name == 'ogbn-mag' else None

        inputs_channels = data[
            'paper'].num_features if dataset_name == 'ogbn-mag' \
            else dataset.num_features

        if torch.cuda.is_available():
            amp = torch.cuda.amp.autocast(enabled=False)
        else:
            amp = torch.cpu.amp.autocast(enabled=args.bf16)
        
        for num_workers in args.num_workers:
            for batch_size in args.eval_batch_sizes:
                if not hetero:
                    subgraph_loader = NeighborLoader(
                        data,
                        num_neighbors=[-1], # layer-wise inference
                        input_nodes=mask,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                    )
                for layers in args.num_layers:
                    if hetero:
                        num_neighbors = [args.hetero_num_neighbors] * layers # batch-wise inference
                        subgraph_loader = NeighborLoader(
                            data,
                            num_neighbors=num_neighbors,
                            input_nodes=mask,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers
                        )
                    else:
                        num_neighbors = [-1] * layers
                    for hidden_channels in args.num_hidden_channels:
                        print('----------------------------------------------')
                        print(f'Batch size={batch_size}, '
                            f'Layers amount={layers}, '
                            f"Num_neighbors={str(num_neighbors).replace(',', ' ')}, "
                            f'Hidden features size={hidden_channels}, '
                            f'Sparse tensor={use_sparse_tensor}, '
                            f'Nr workers={num_workers}')
                        params = {
                            'inputs_channels': inputs_channels,
                            'hidden_channels': hidden_channels,
                            'output_channels': num_classes,
                            'num_heads': args.num_heads,
                            'num_layers': layers,
                        }

                        model = get_model(
                            model_name, params,
                            metadata=data.metadata() if hetero else None)
                        model = model.to(device)
                        model.eval()
                        if use_cpu_affinity:
                            with subgraph_loader.enable_cpu_affinity(loader_cores=args.loader_cores, compute_cores=args.compute_cores):
                                with amp:
                                    for _ in range(args.warmup):
                                        print(f"WARMUP TIME")
                                        with timeit(): 
                                            try:
                                                model.inference(subgraph_loader, device,
                                                                progress_bar=True)
                                            except RuntimeError:
                                                pass
                                    print("INFERENCE TIME")
                                    with timeit():
                                        try:
                                            model.inference(subgraph_loader, device,
                                                            progress_bar=True)
                                        except RuntimeError:
                                            pass
                        else:
                            with amp:
                                    for _ in range(args.warmup):
                                        print(f"WARMUP TIME")
                                        with timeit(): 
                                            try:
                                                model.inference(subgraph_loader, device,
                                                                progress_bar=True)
                                            except RuntimeError:
                                                pass
                                    print("INFERENCE TIME")
                                    with timeit():
                                        try:
                                            model.inference(subgraph_loader, device,
                                                            progress_bar=True)
                                        except RuntimeError:
                                            pass
                            


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN inference benchmark')
    argparser.add_argument('--datasets', nargs='+',
                           default=[], type=str)
    argparser.add_argument(
        '--use-sparse-tensor',default=0, type=int,
        help='use torch_sparse.SparseTensor as graph storage format')
    argparser.add_argument(
        '--models', nargs='+',
        default=['gcn'], type=str)
    argparser.add_argument('--root', default='../../data', type=str,
                           help='relative path to look for the datasets')
    argparser.add_argument('--eval-batch-sizes', nargs='+',
                           default=[512, 1024, 2048, 4096, 8192], type=int)
    argparser.add_argument('--num-layers', nargs='+', default=[2, 3], type=int)
    argparser.add_argument('--num-hidden-channels', nargs='+',
                           default=[64, 128, 256], type=int)
    argparser.add_argument(
        '--num-heads', default=2, type=int,
        help='number of hidden attention heads, applies only for gat and rgat')
    argparser.add_argument(
        '--hetero-num-neighbors', default=10, type=int,
        help='number of neighbors to sample per layer for hetero workloads')
    argparser.add_argument('--num-workers', nargs='+', default=[2], type=int)
    argparser.add_argument('--warmup', default=1, type=int)
    argparser.add_argument('--profile', action='store_true')
    argparser.add_argument('--bf16', action='store_true')
    argparser.add_argument('--cpu-affinity', default=1, type=int)
    argparser.add_argument('--loader-cores', nargs='+', default=[], type=int)
    argparser.add_argument('--compute-cores', nargs='+', default=[], type=int)

    args = argparser.parse_args()

    run(args)
