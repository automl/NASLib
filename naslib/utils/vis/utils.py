import logging
import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import seaborn as sns

logger = logging.getLogger(__name__)

def plot_architectural_weights(config, optimizer):
    all_weights = torch.load(f'{config.save}/arch_weights.pt') # load alphas

    # unpack search space information
    alpha_dict = {}
    min_soft, max_soft = np.inf, -np.inf
    for graph in optimizer.graph._get_child_graphs(single_instances=True):
        for edge_weights, (u, v, edge_data) in zip(all_weights, graph.edges.data()):

            if edge_data.has("alpha"):
                total_steps, num_alphas = edge_weights.shape
                steps_per_epoch = total_steps // config.search.epochs
                disc_weights = torch.mean(edge_weights.detach().reshape(-1, steps_per_epoch, num_alphas), axis=1).cpu()    
                soft_weights = torch.softmax(disc_weights, dim=-1).numpy()

                cell_name = edge_data['cell_name'] if hasattr(edge_data, 'cell_name') else ""
                alpha_dict[(u, v, cell_name)] = {}
                alpha_dict[(u, v, cell_name)]['op_names'] = [op.get_op_name for op in edge_data.op.get_embedded_ops()]
                alpha_dict[(u, v, cell_name)]['alphas'] = soft_weights

                min_soft = min(min_soft, np.min(soft_weights))
                max_soft = max(max_soft, np.max(soft_weights))

    max_rows = 4 # plot heatmaps in increments of n_rows edges
    for start_id in range(0, len(alpha_dict.keys()), max_rows):
        
        # calculate number of rows in plot
        n_rows = min(max_rows, len(alpha_dict.keys())-start_id)
        logger.info(f"Creating plot {config.save}/arch_weights_{start_id+1}to{start_id+n_rows}.png")

        # define figure and axes and NASLib colormap
        fig, axes = plt.subplots(nrows=n_rows, figsize=(10, max_rows))
        cmap = sns.diverging_palette(230, 0, 90, 60, as_cmap=True)

        # iterate over arch weights and create heatmaps
        for ax_id, (u, v, cell_name) in enumerate(list(alpha_dict.keys())[start_id:start_id+n_rows]): 
            map = sns.heatmap(
                alpha_dict[u, v, cell_name]['alphas'].T,
                cmap=cmap, 
                vmin=min_soft,
                vmax=max_soft, 
                ax=axes[ax_id],
                cbar=True
            )

            op_names = alpha_dict[(u, v, cell_name)]['op_names']

            if ax_id < n_rows-1:        
                axes[ax_id].set_xticks([])
            axes[ax_id].set_ylabel(f"{u, v}", fontdict=dict(fontsize=6))
            axes[ax_id].set_yticks(np.arange(len(op_names)) + 0.5)
            fontsize = max(6, 40/len(op_names))
            axes[ax_id].set_yticklabels(op_names, rotation=360, fontdict=dict(fontsize=fontsize))
            if cell_name != "":
                axes[ax_id].set_title(cell_name, fontdict=dict(fontsize=6))
            cbar = map.collections[0].colorbar
            cbar.ax.tick_params(labelsize=6)
            cbar.ax.set_title('softmax', fontdict=dict(fontsize=6))

        axes[ax_id].xaxis.set_tick_params(labelsize=6)
        axes[ax_id].set_xlabel("Epoch", fontdict=dict(fontsize=6))

        fig.suptitle(f"optimizer: {config.optimizer}, search space: {config.search_space}, dataset: {config.dataset}, seed: {config.seed}")
        fig.tight_layout()
        fig.savefig(f"{config.save}/arch_weights_{start_id+1}to{start_id+n_rows}.png", dpi=300)

