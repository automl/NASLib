import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    X = torch.load("save_arch/tensor.pt")
    X = X.numpy()

    for idx, x in enumerate(X):
        sns.heatmap(x.T, vmax=np.max(X), vmin=np.min(X))
        plt.title(f"arch weights for operation {idx}")
        plt.xlabel("steps")
        plt.ylabel("alpha values")
        plt.savefig(f"heatmap_{idx}.png")
        plt.cla()
        plt.clf()
        plt.close()