import logging
import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import seaborn as sns

logger = logging.getLogger(__name__)

def plot_architectural_weights(config, optimizer):
    # load alphas
    arch_weights = torch.load(f'{config.save}/arch_weights.pt')

    # discretize and softmax alphas
    for i, edge_weights in enumerate(arch_weights):
        total_steps, num_alphas = edge_weights.shape
        num_epochs = config.search.epochs
        steps_per_epoch = total_steps // num_epochs
        
        disc_weights = torch.mean(edge_weights.detach().reshape(-1, steps_per_epoch, num_alphas), axis=1).cpu()    
        arch_weights[i] = torch.softmax(disc_weights, dim=-1).numpy()

    # define diverging colormap with NASLib colors
    cmap = sns.diverging_palette(230, 0, 90, 60, as_cmap=True)

    # unpack search space information
    edge_names, op_names = [], []
    for graph in optimizer.graph._get_child_graphs(single_instances=True):
        for u, v, edge_data in graph.edges.data():
            if edge_data.has("alpha"):
                edge_names.append((u, v))
                op_names.append([op.get_op_name for op in edge_data.op.get_embedded_ops()])

    # define figure and axes
    fig, axes = plt.subplots(nrows=len(arch_weights), figsize=(10, np.array(op_names).size/10))
    cax = fig.add_axes([.95, 0.12, 0.0075, 0.795])
    cax.tick_params(labelsize=6)
    cax.set_title('alphas', fontdict=dict(fontsize=6))

    # unpack number of epochs
    num_epochs = config.search.epochs

    # iterate over arch weights and create heatmaps
    for (i, edge_weights) in enumerate(arch_weights):
        num_steps, num_alphas = edge_weights.shape
        
        sns.heatmap(
            edge_weights.T,
            cmap=cmap, 
            vmin=np.min(arch_weights),
            vmax=np.max(arch_weights), 
            ax=axes[i],
            cbar=True,
            cbar_ax=cax
        )

        if i == len(arch_weights) - 1:
            # axes[i].set_xticks(np.arange(stop=num_steps+num_steps/num_epochs, step=num_steps/num_epochs)) 
            # axes[i].set_xticklabels(np.arange(num_epochs+1), rotation=360, fontdict=dict(fontsize=6))
            axes[i].xaxis.set_tick_params(labelsize=6)
            axes[i].set_xlabel("Epoch", fontdict=dict(fontsize=6))
        else:
            axes[i].set_xticks([])
        
        axes[i].set_ylabel(edge_names[i], fontdict=dict(fontsize=6))
        axes[i].set_yticks(np.arange(num_alphas) + 0.5)
        axes[i].set_yticklabels(op_names[i], rotation=360, fontdict=dict(fontsize=5))

    fig.tight_layout(rect=[0, 0, 0.95, 0.925], pad=0.25)
    
    _, search_space, dataset, optimizer, seed = config.save.split('/')
    fig.suptitle(f"optimizer: {optimizer}, search space: {search_space}, dataset: {dataset}, seed: {seed}")

    fig.savefig(f"{config.save}/arch_weights.pdf", dpi=300)

