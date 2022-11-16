import pandas as pd
import os
import plotly.subplots as ps
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
from os import listdir
import re

SUPPORTED_SETS = {
    'rgcn':'ogbn-mag',
    'gat':'ogbn-products',
    'gcn':'ogbn=products'
}

KEYS = ['TEST_ID', 'T_inf', 'T_warm', 'BATCH', 'LAYERS', 'NUM_NBRS', 'FEAT', 'ST', 'NR_WORK'] 
CONFIG = ['MODEL', 'DATASET', 'HYPERTHREADING', 'COMP_AFFINITY', 'DL_AFFINITY', 'OMP_NUM_THREADS','GOMP_CPU_AFFINITY','OMP_PROC_BIND']    
def load_logfile(file):
    data = {k: [] for k in KEYS+CONFIG}

    with open(file, 'r') as f:
        lines = f.readlines()
        it = 0
        end_of_config = False
        for line in lines:
            line = line.rstrip('\n')
            if not end_of_config:
                for conf in CONFIG:
                    if conf in line:
                        val = line.split(':')[1].strip() if line.split(':')[1] != ' ' else None
                        data[conf].append(val)
                caff = re.search('CAFF(.*).log', file)
                data['COMP_AFFINITY'].append(caff.group(1) if caff else 0)
                
            if line == 'BENCHMARK STARTS':
                end_of_config = True
                
            elif "----" in line:
                it += 1
                data['TEST_ID'].append(it)
                warmup = False
            elif 'Batch' in line:
                results = line.split(',')
                for i, key in enumerate(KEYS[3:]):
                    data[key].append(results[i].split('=')[1])
            elif 'WARMUP' in line:
                warmup = True
            elif 'INFERENCE' in line:
                warmup = False
            elif 'Time' in line:
                time = float(line.rstrip('s').split(":")[1].lstrip())
                data['T_warm'].append(time) if warmup else data['T_inf'].append(time)
        print('EOF')

        for conf in CONFIG:
            data[conf] = [data[conf][0]]*len(data['TEST_ID'])        
        return data
            
def load_data(directory):
    list_of_dicts = []
    for file in listdir(directory):
        if ".log" in file:
            list_of_dicts.append(load_logfile(f'{directory}/{file}'))
    result_dict = {k:[] for k in list_of_dicts[0].keys()} 
    for d in list_of_dicts:
        for k, v in d.items():
            for i in v:
                result_dict[k].append(i)            
    result_dict.pop('TEST_ID')
    table = pd.DataFrame(result_dict)
    table['NR_WORK'] = table['NR_WORK'].astype(int)
    table['OMP_PROC_BIND'] = table['OMP_PROC_BIND'].astype(str)
    table['COMP_AFFINITY'] = table['COMP_AFFINITY'].astype(int)
    table['DL_AFFINITY'] = table['DL_AFFINITY'].astype(int)
    #table.sort_values(by=['MODEL','NR_WORK','DL_AFFINITY'], inplace=True)
    table.to_csv(f"{directory}/{directory.split('/')[-1]}.csv", na_rep='-', index_label="TEST_ID", header=True)
    print(f'Finished processing logs in folder {directory}')               
    return table

def plot_grid(data, ht, feat_size, time, sparse_tensor):

    # plotting 
    affinity_setups=['Baseline','DL','DL+C1','DL+C2','DL+C3','DL+C4']
    colors=list(px.colors.qualitative.Plotly)[:len(affinity_setups)]
    color_dict = {k:v for k,v in zip(affinity_setups,colors)}
    layers=[2,3]
    batches=[512,1024,2048,4096,8192]
    fig = ps.make_subplots(rows=2, cols=5, 
                           shared_xaxes=False, shared_yaxes=False,
                           subplot_titles=[f"Layers={l}, Feat size={feat_size}, Batch={b}" for l in layers for b in batches])    
    once = True
    for r, layer in enumerate(layers):
        for c, batch in enumerate(batches):
            for s in list(np.unique(data["setup"])):
                sample = data.loc[(data['BATCH'] == str(batch)) & (data['LAYERS'] == str(layer)) & (data['setup'] == s)]
                sample = sample.sort_values(by='NR_WORK')
                #print(sample[['T_inf', 'T_warm']].mean()) #TODO: worth checking why sometimes warmup time < inference
                if feat_size == 128 and s == 'DL' and sparse_tensor == False: #workaround for doubled datapoints
                    sample = sample.iloc[::2, :]
                if time == 'warm':
                    y = sample['T_warm']
                elif time == 'inf':
                    y = sample['T_inf']
                elif time == 'mean':
                    y = (sample['T_inf'] + sample['T_inf'])/2
                fig.add_trace(
                    go.Scatter(x=sample["NR_WORK"], y=y, 
                               line=dict(color=color_dict.get(s)),
                               name=s,
                               showlegend=once,
                               ),
                    row=r+1, col=c+1)
            once = False

    ht_label = 'ON' if int(ht) == 1 else 'OFF'
    fig.update_layout(height=600, width=1800, 
                      title_text=f"2xICX + 512GB RAM, Model = gcn, dataset = ogbn_products, sparse_tensor={sparse_tensor}, Hyperthreading {ht_label}")
    fig.update_xaxes(type='category', categoryarray=np.unique(data["NR_WORK"]))
    fig.update_yaxes(dtick=10)
    fig.add_annotation(
                    text=
                    f"""
                    x-axis - number of DL Workers
                    y-axis - {time} time (s) 
                    """, 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=-0.15,
                    bordercolor='black',
                    font=dict(size=14),
                    borderwidth=1)            
    figname=f"{plotdir}/feat{feat_size}HT{ht}.png"
    print(f"Created {figname}")
    fig.write_image(figname)
    
         
def model_mask(data):
    data['setup'] = np.nan
    data = data.assign(setup=np.where((data['COMP_AFFINITY'] == 0) & (data['DL_AFFINITY'] == 0), 'Baseline', data['setup'])) 
    data = data.assign(setup=np.where((data['COMP_AFFINITY'] == 0) & (data['DL_AFFINITY'] == 1), 'DL', data['setup']))
    data = data.assign(setup=np.where((data['COMP_AFFINITY'] == 1) & (data['DL_AFFINITY'] == 1), 'DL+C1', data['setup']))
    data = data.assign(setup=np.where((data['COMP_AFFINITY'] == 2) & (data['DL_AFFINITY'] == 1), 'DL+C2', data['setup']))
    data = data.assign(setup=np.where((data['COMP_AFFINITY'] == 3) & (data['DL_AFFINITY'] == 1), 'DL+C3', data['setup']))
    data = data.assign(setup=np.where((data['COMP_AFFINITY'] == 4) & (data['DL_AFFINITY'] == 1), 'DL+C4', data['setup']))
    return data
    
    
    
if __name__ == '__main__':
        
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    CWD = os.getcwd()
    print("Current working directory: {0}".format(CWD))

    #CWD=f'pytorch_geometric/benchmark/inference/logs/{platform}'
    feat_size = [16, 128]
    sparse_tensor = False
    oper = 'spmm' if sparse_tensor else 'scatteradd'
    hyperthreading = ['0','1']
    measured_time = "mean"
    
    for ht in hyperthreading:
        for fs in feat_size:
            logdir= f"{CWD}/logs-all/{oper}/logs-feat{fs}" if sparse_tensor else f"{CWD}/logs-all/{oper}"
            plotdir=f'{logdir}/plots'
            os.makedirs(plotdir, exist_ok=True)
            if sparse_tensor:
                baseline_data = load_data(f'{logdir}/baseline')
                affinity_data = load_data(f'{logdir}/dl-affinity')
                baseline = baseline_data.loc[(baseline_data['ST'] == 'True') & (baseline_data['HYPERTHREADING'] == ht) & (baseline_data['FEAT'] == str(fs))]
                aff = affinity_data.loc[(affinity_data['HYPERTHREADING'] == ht) & (affinity_data['OMP_PROC_BIND'] == 'None') ]
                data = pd.concat([baseline, aff])
                data = model_mask(data)
                plot_grid(data, ht, fs, measured_time, sparse_tensor)
                
            else:
                scatter_data = load_data(f'{CWD}/logs-all/{oper}/logs')
                scatter_data = scatter_data.loc[(scatter_data['HYPERTHREADING'] == ht) & (scatter_data['FEAT'] == str(fs))]
                scatter_data = model_mask(scatter_data)
                plot_grid(scatter_data, ht, fs, measured_time, sparse_tensor)
    print('END')
