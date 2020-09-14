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


def get_trajectories(opt_dict, methods=['RE', 'RS', 'TPE']):
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

