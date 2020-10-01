import os
import json
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed


colors={
        'BOHB_joint': 'darkorange',
        'BOHB_nas': 'dodgerblue',
        'RE': 'crimson',
		'RS': 'dodgerblue',
		'RL': 'sienna',
		'TPE': 'deepskyblue',
        'PC-DARTS': 'darkorange',
        'True': 'darkorange',
        'Surrogate': 'deepskyblue',
        'GDAS': 'crimson',
        'DARTS': 'dodgerblue',
        'RandomNAS': 'crimson',
        'HB': 'darkgray',
        'BOHB': 'gold'
}

markers={
        'BOHB_joint': '^',
        'BOHB_nas': 'v',
        'RS': 'D',
		'RE': 'o',
		'GDAS': 's',
		'RL': 's',
        'True': '^',
		'Surrogate': 'h',
        'PC-DARTS': '^',
		'DARTS': 'h',
		'RandomNAS': 's',
        'HB': '>',
        'BOHB': '*',
        'TPE': '<'
}


def get_trajectories(opt_dict, methods=['RE', 'RS']):
    all_trajectories = {}

    for m in methods:
        dfs = []
        data = opt_dict[m]
        losses = data['losses']
        times = data['time']
        for i in range(len(losses)):
            loss = losses[i]
            time = times[i]
            print('Seed: ', i, ' MIN: ', min(loss))
            df = pd.DataFrame({str(i): loss}, index=time)
            dfs.append(df)

        df = merge_and_fill_trajectories(dfs, default_value=None)
        if df.empty:
            continue
        print(m, df.shape)

        all_trajectories[m] = {
            'time_stamps': np.array(df.index),
            'losses': np.array(df.T)
        }

    return all_trajectories


def merge_and_fill_trajectories(pandas_data_frames, default_value=None):
	# merge all tracjectories keeping all time steps
	df = pd.DataFrame().join(pandas_data_frames, how='outer')

	# forward fill to make it a propper step function
	df=df.fillna(method='ffill')

	if default_value is None:
	# backward fill to replace the NaNs for the early times by
	# the performance of a random configuration
		df=df.fillna(method='bfill')
	else:
		df=df.fillna(default_value)

	return(df)


def plot_losses(fig, ax, axins, incumbent_trajectories, regret=True,
                incumbent=None, show=True, linewidth=3, marker_size=10,
                xscale='log', xlabel='wall clock time [s]', yscale='log',
                ylabel=None, legend_loc = 'best', xlim=None, ylim=None,
                plot_mean=True, labels={}, markers=markers, colors=colors,
                figsize=(16,9)):

    if regret:
        if ylabel is None: ylabel = 'regret'
		# find lowest performance in the data to update incumbent

        if incumbent is None:
            incumbent = np.inf
            for tr in incumbent_trajectories.values():
                incumbent = min(tr['losses'][:,-1].min(), incumbent)
            print('incumbent value: ', incumbent)

    for m,tr in incumbent_trajectories.items():
        trajectory = np.copy(tr['losses'])
        if (trajectory.shape[0] == 0): continue
        if regret: trajectory -= incumbent

        sem  =  np.sqrt(trajectory.var(axis=0, ddof=1)/tr['losses'].shape[0])
        if plot_mean:
            mean =  trajectory.mean(axis=0)
        else:
            mean = np.median(trajectory,axis=0)
            sem *= 1.253

        ax.fill_between(tr['time_stamps'], mean-2*sem, mean+2*sem,
                        color=colors[m], alpha=0.2)

        ax.plot(tr['time_stamps'],mean,
                label=labels.get(m, m), color=colors.get(m, None),linewidth=linewidth,
                marker=markers.get(m,None), markersize=marker_size, markevery=(0.1,0.1))

        if axins is not None:
            axins.plot(tr['time_stamps'],mean,
                       label=labels.get(m, m), color=colors.get(m, None),linewidth=linewidth,
                       marker=markers.get(m,None), markersize=marker_size, markevery=(0.1,0.1))

    return (fig, ax)

import networkx as nx
import matplotlib.pyplot as plt
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import Identity

def plot_cells():
    cell = Graph()
    cell.add_nodes_from(range(1, 8))
    cell.add_edge(1, 3, op="sep_conv_3x3")
    cell.add_edge(1, 4, op="identity")
    cell.add_edge(1, 5, op="identity")
    cell.add_edge(1, 6, op="identity")
    cell.add_edge(2, 3, op="sep_conv_3x3")
    cell.add_edge(2, 6, op="sep_conv_3x3")
    cell.add_edge(3, 4, op="identity")
    cell.add_edge(2, 5, op="dil_conv_5x5")
    cell.add_edges_from([(i, 7) for i in range(3, 7)])

    redu = Graph()
    redu.add_nodes_from(range(1, 8))
    redu.add_edge(1, 3, op="max_pool_3x3")
    redu.add_edge(1, 4, op="max_pool_3x3")
    redu.add_edge(1, 5, op="max_pool_3x3")
    redu.add_edge(2, 3, op="max_pool_3x3")
    redu.add_edge(2, 4, op="max_pool_3x3")
    redu.add_edge(2, 5, op="identity")
    redu.add_edge(3, 6, op="identity")
    redu.add_edge(4, 6, op="identity")
    redu.add_edges_from([(i, 7) for i in range(3, 7)])


    fig, (ax_top, ax_bot) = plt.subplots(nrows=2, ncols=1)

    pos = {
        1: [-1, .5],
        2: [-1, -.5],
        3: [0, .35],
        4: [.6, .5],
        5: [0, -.5],
        6: [.5, 0],
        7: [1, 0]
    }
    nx.draw_networkx_nodes(cell, pos, ax=ax_top, node_color=['g', 'g', 'y', 'y', 'y', 'y', 'm'])

    nx.draw_networkx_labels(cell, pos, {k: str(k) for k in cell.nodes()}, ax=ax_top)
    nx.draw_networkx_edges(cell, pos, ax=ax_top)
    nx.draw_networkx_edge_labels(cell, pos, 
        {(u, v): d.op for u, v, d in cell.edges(data=True) if not isinstance(d.op, Identity)},
        label_pos=.68, 
        ax=ax_top,
        bbox=dict(facecolor='white', alpha=0.4, edgecolor='white'),
        font_size=10
        )
    ax_top.set_title("Normal cell")


    pos = {
        1: [-1, 1],
        2: [-1, -1],
        3: [0, 1],
        4: [0, -.2],
        5: [0, -1],
        6: [.55, .1],
        7: [1, 0]
    }
    nx.draw_networkx_nodes(redu, pos, ax=ax_bot, node_color=['g', 'g', 'y', 'y', 'y', 'y', 'm'])

    nx.draw_networkx_labels(redu, pos, {k: str(k) for k in redu.nodes()}, ax=ax_bot)
    nx.draw_networkx_edges(redu, pos, ax=ax_bot)
    nx.draw_networkx_edge_labels(redu, pos, 
        {(u, v): d.op for u, v, d in redu.edges(data=True) if not isinstance(d.op, Identity)},
        label_pos=.68, 
        ax=ax_bot,
        bbox=dict(facecolor='white', alpha=0.4, edgecolor='white'),
        font_size=10
    )
    ax_bot.set_title("Reduction cell")
    plt.tight_layout()
    plt.savefig('darts_cells.pdf')

    print()

from naslib.utils import utils
from naslib.search_spaces import DartsSearchSpace
from naslib.optimizers import DARTSOptimizer
import torch

def params_from_checkpoint():

    model = torch.load('darts/model_final.pth', map_location=torch.device('cpu'))['model']
    print(np.sum(np.prod(v.size()) for v in model.values()) / 1e6, "M")


if __name__ == '__main__':
    #plot_cells()
    params_from_checkpoint()
