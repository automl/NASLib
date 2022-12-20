import logging
import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import seaborn as sns

logger = logging.getLogger(__name__)

def plot_architectural_weights(config):
    arch_weights = torch.load(f'{config.save}/arch_weights.pt')
    
    for i, edge_weights in enumerate(arch_weights):
        arch_weights[i] = torch.softmax(edge_weights.detach(), dim=1).cpu().numpy()

    num_epochs = config.search.epochs
    cmap = sns.diverging_palette(230, 0, 90, 60, as_cmap=True)

    fig, axes = plt.subplots(nrows=len(arch_weights))
    cax = fig.add_axes([.9, 0.05, .0125, 0.925])

    for i, edge_weights in enumerate(arch_weights):
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
            axes[i].set_xticks(np.arange(stop=num_steps+num_steps/num_epochs, step=num_steps/num_epochs)) 
            axes[i].set_xticklabels(np.arange(num_epochs+1), rotation=360, fontdict=dict(fontsize=6))
        else:
            axes[i].set_xticks([])

        axes[i].set_ylabel('edge', fontdict=dict(fontsize=6))
        axes[i].set_yticks(np.arange(num_alphas))
        axes[i].set_yticklabels(['op'] * num_alphas, rotation=360, fontdict=dict(fontsize=6))

    fig.tight_layout(rect=[0, 0, 0.9, 1], pad=0.5)
    fig.savefig(f"{config.save}/arch_weights.png", dpi=300)
    plt.close()

