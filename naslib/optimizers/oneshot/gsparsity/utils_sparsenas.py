import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model_search_multipath import Network as SearchNetwork
from graphviz import Digraph

matplotlib.use("Agg")

def get_regularization_term(model_params, args):
    reg_loss = 0
    for group in model_params:
        if group["label"] == "unprunable": # weights from ops like stem, cell preprocessing and classifier.
            continue
            
        if group['weight_decay'] is None:
            continue
            
        group_norm = torch.zeros(1).cuda()
        group_dim = (torch.zeros(1)).cuda()
        for x in group['params']:
            group_norm += torch.norm(x)**2
            group_dim += torch.numel(x)
        group_norm = torch.sqrt(group_norm)

#         print("group name: {}, dim {} (sqrt_dim {})".format(group['op_name'], group_dim.item(), group_dim.item() ** args.normalization_exponent))
        
        if args.normalization == "mul":
            reg_loss += group_norm * torch.pow(group_dim, args.normalization_exponent)
        elif args.normalization == "div":
            reg_loss += group_norm / torch.pow(group_dim, args.normalization_exponent)
        else:
            reg_loss += group_norm

    return reg_loss

def plot_individual_op_norm(model, filename, normalization="none", normalization_exponent=0):
    """plot the norm of each operation in the given model"""
    
    op_norms = [torch.norm(p) for p in model.parameters() if p.requires_grad]
    op_names = [q for q, p in model.named_parameters() if p.requires_grad]
    op_sizes = [p.numel() for p in model.parameters() if p.requires_grad]
        
    num_ops = len(op_names)
    f = plt.figure(num=None, figsize=(num_ops*0.15, 6), dpi=100, facecolor='w', edgecolor='k')
    if normalization == "none":
        for i, op_norm in enumerate(op_norms):
            plt.semilogy(i, op_norm.item(),"o")
    elif normalization == "mul":
        for i, (op_norm, op_size) in enumerate(zip(op_norms, op_sizes)):
            op_norm_normalized = op_norm * (op_size ** normalization_exponent)
            plt.semilogy(i, op_norm_normalized.item(),"o")
    elif normalization == "div":
        for i, (op_norm, op_size) in enumerate(zip(op_norms, op_sizes)):
            op_norm_normalized = op_norm / (op_size ** normalization_exponent)
            plt.semilogy(i, op_norm_normalized.item(),"o")

    plt.xticks(np.arange(num_ops), op_names, rotation=90)
    plt.xlim(-1, num_ops)    
    plt.ylim(1e-5, 1e5)    
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def acc_n_loss(train_loss, test_acc, filename, train_acc=None, test_loss=None, train_loss_reg=None):
    if train_acc is not None and test_loss is not None:
        fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.2))
        fig.suptitle('Loss and Acc')
        axs[0,0].semilogy(train_loss, label='loss')

        if train_loss_reg is not None:
            axs[0,0].semilogy(train_loss_reg, label='loss+reg')
            
        axs[0,0].grid(True)
        axs[0,0].set_xlabel('Epochs')
        axs[0,0].set_ylabel('Training loss')

        axs[0,1].plot(train_acc)
        axs[0,1].grid(True)
        axs[0,1].set_ylim(0,101)
        axs[0,1].set_yticks(np.arange(0, 101, 5))
        axs[0,1].set_xlabel('Epochs')
        axs[0,1].set_ylabel('Train accuracy (in %)')

        axs[1,0].semilogy(test_loss)
        axs[1,0].grid(True)
        axs[1,0].set_xlabel('Epochs')
        axs[1,0].set_ylabel('Test loss')

        axs[1,1].plot(test_acc)
        axs[1,1].grid(True)
        axs[1,1].set_ylim(0,101)
        axs[1,1].set_yticks(np.arange(0, 101, 5))
        axs[1,1].set_xlabel('Epochs')
        axs[1,1].set_ylabel('Test accuracy (in %)')

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)       
        
    elif train_acc is not None and test_loss is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))
        fig.suptitle('Loss and Acc')
        ax1.semilogy(train_loss)
        ax1.grid(True)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training loss')

        ax2.plot(train_acc)
        ax2.grid(True)
        ax2.set_ylim(0,101)
        ax2.set_yticks(np.arange(0, 101, 5))
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Train accuracy (in %)')

        ax3.plot(test_acc)
        ax3.grid(True)
        ax3.set_ylim(0,101)
        ax3.set_yticks(np.arange(0, 101, 5))
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Test accuracy (in %)')

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        
    elif train_acc is None and test_loss is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 4.8))
        fig.suptitle('Loss and Acc')
        ax1.semilogy(train_loss)
        ax2.plot(test_acc)
        ax1.grid(True)
        ax2.grid(True)
        #ax1.set_ylim(bottom=0)
        ax2.set_ylim(0,101)
        ax2.set_yticks(np.arange(0, 101, 5))
        ax1.title.set_text('Loss')
        ax1.set_xlabel('Epochs')
        ax2.set_xlabel('Epochs')
        ax1.set_ylabel('Training loss')
        ax2.set_ylabel('Test accuracy (in %)')
        ax2.title.set_text('Accuracy')
        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

        
def group_model_params_by_cell(model, network, mu=None):
  """
  This functions put the same operation of different cells into the same vector (the group will be passed to optimizer).
    
  One operation may consist of several suboperations. For example, op 3 of edge 7 consists of (_ops.7._ops.3.op.1.weight, _ops.7._ops.3.op.2.weight, _ops.7._ops.3.op.5.weight, _ops.7._ops.3.op.6.weight). Op 3 of edge 7 from all cells of the same type (normal or reduce) will be grouped as one, "_ops.7._ops.3". During discretization, the operation "_ops.7._ops.3" will be pruned if its norm is smaller than the pruning threshold.
  """
    
  assert network.num_ops <=9, "The number of operations should be smaller than 10 (but got {}).".format(network.num_ops)
  assert network.num_edges <=100, "The number of edges should be smaller than 100 (but got {}).".format(network.num_edges)
  
  """
  The operations that are trainable but not prunable should be separated from trainable and prunable operations. "Unprunable" means these are the operations that will be definitely kept in the final network (such as the preprocessing layer, the final classifier layer, and the preprocessing of input nodes in each cell), in contrast to the operations that may be pruned away after searching is completed.
  """    
  ops_unprunable = []
  ops_unscale = []
  for op in model.stem:
        for param in op.parameters():
            ops_unprunable.append(param)
            ops_unscale.append(False)
                
  """model.global_pooling is before classifier, but it is not trainable"""
  classifier_weight, classifier_bias = model.classifier.parameters()
  ops_unprunable.extend([classifier_weight, classifier_bias])
  ops_unscale.append(False)
  
  """The operations that are prunable are put in a separate dictionary."""
  ops_prunable_normal = dict()
  ops_prunable_reduce = dict()
  op_is_scale_normal = dict()
  op_is_scale_reduce = dict()
  for edge in range(network.num_edges):
    for op in range(network.num_ops):
        ops_prunable_normal["_ops.{}._ops.{}".format(edge, op)] = []
        ops_prunable_reduce["_ops.{}._ops.{}".format(edge, op)] = []
        op_is_scale_normal["_ops.{}._ops.{}".format(edge, op)] = []
        op_is_scale_reduce["_ops.{}._ops.{}".format(edge, op)] = []
        
  for cell_index, m in enumerate(model.cells):
        op_index = 0
        edge_index = 0
        
        for name, param in m.named_parameters():
#             print("op_name: {}".format(name), end=" ")
            """
            An example of "name" is _ops.8._ops.4.op.2.weight, where 8 represents the edge, 4 is the op index, and 2 is the subop of op 4 (op 4 consists of several subops).
            """
            if "_ops" in name:
                if "_ops.0._ops.0" in name: # beginning of a new cell
                    cur_op_name = name[0:13] # assuming the number of cells < 10
                    pre_op_name = cur_op_name
                else:
                    if edge_index <= 9:
                        cur_op_name = name[0:13] #example: extract "_ops.3._ops.4" from "_ops.3._ops.4.op.2.weight"
                    else:
                        cur_op_name = name[0:14] #example: extract "_ops.13._ops.4" from "_ops.13._ops.4.op.2.weight"
                
                if cur_op_name == pre_op_name: #still the same op
                    pass
                else: # current op is a new op
                    op_index += 1
                    if op_index == network.num_ops: # the current op belongs to a new edge
                        new_edge = True
                        op_index = 0
                        edge_index += 1
                    else: # still the same edge
                        new_edge = False
                        pre_op_name = cur_op_name
                        
                    if edge_index <= 9: # get the name of the current (new) op
                        cur_op_name = name[0:13]
                    else:
                        cur_op_name = name[0:14]
                           
                    if new_edge:
                        pre_op_name = cur_op_name
                        
#                 print("  name is      {}, edge index is {}, op index is {}".format(name, edge_index, op_index))
#                 print("  cur_op_name: {}".format(cur_op_name))
#                 print("  pre_op_name: {}".format(pre_op_name))
                
                if cell_index in network.reduce_cell_indices:
                    ops_prunable_reduce[cur_op_name].append(param)
                    op_is_scale_reduce[cur_op_name].append("scale" in name)
#                     print("is a scale op? {}".format(op_is_scale_reduce[cur_op_name][-1]))
                else:
                    ops_prunable_normal[cur_op_name].append(param)
                    op_is_scale_normal[cur_op_name].append("scale" in name)
#                     print("is a scale op? {}".format(op_is_scale_normal[cur_op_name][-1]))
            else:
                ops_unprunable.append(param)
                ops_unscale.append(False)
#                 print("is a scale op? {}".format(ops_unscale[-1]))

                
  """
  define the parameter groups that will be passed to the optimizer. Prunable operations will have a nonzero weight decay (mu), while nonprunable operations do not have a mu.
  """
  model_params = []
  for op_name, op_param in ops_prunable_normal.items():
    model_params.append(dict(params=op_param, label="normal", op_name=op_name, weight_decay=mu, scale=op_is_scale_normal[op_name]))
  for op_name, op_param in ops_prunable_reduce.items():
    model_params.append(dict(params=op_param, label="reduce", op_name=op_name, weight_decay=mu, scale=op_is_scale_reduce[op_name]))
  model_params.append(dict(params=ops_unprunable, label="unprunable", op_name="unprunable", weight_decay=None, scale=ops_unscale))
  
  return model_params



def compute_op_norm_across_cells(model_params):
    # compute the norm of the vector containing the weights of the same operation in different cells (e.g., sep_conv_3x3)
    # normal cells and reduction cells are computed separately
    op_norm_normal_dict = {}
    op_norm_reduce_dict = {}
    for operation in model_params:
        if operation["label"] == "unprunable": # weights from ops like stem, cell preprocessing and classifier.
            continue
        
        params = operation["params"]
        params_norm_square = 0
        params_size = 0
        for param in params:
            params_norm_square += torch.norm(param) ** 2 
            params_size += param.numel()
         
        if operation["label"] == "normal":
            op_norm_normal_dict[operation["op_name"]] = (torch.sqrt(params_norm_square), params_size) # take the square root to get the L2 norm
        elif operation["label"] == "reduce":
            op_norm_reduce_dict[operation["op_name"]] = (torch.sqrt(params_norm_square), params_size)

    return op_norm_normal_dict, op_norm_reduce_dict

def plot_op_norm_across_cells(model_params, filename, normalization="none", normalization_exponent=0):
    op_norm_normal_dict, op_norm_reduce_dict = compute_op_norm_across_cells(model_params)
        
    num_ops = len(op_norm_normal_dict)
    f1 = plt.figure(num=None, figsize=(num_ops*0.15, 6), dpi=100, facecolor='w', edgecolor='k')    
    op_names = []
    if normalization == "none":
        for i, (op_name, (op_norm, _)) in enumerate(op_norm_normal_dict.items()):
            op_names.append(op_name)
            plt.semilogy(i, op_norm.item(), "o")
    elif normalization == "mul":
        for i, (op_name, (op_norm, op_size)) in enumerate(op_norm_normal_dict.items()):
            op_names.append(op_name)
            op_norm_normalized = op_norm * (op_size ** normalization_exponent)
            plt.semilogy(i, op_norm_normalized.item(), "o")
    elif normalization == "div":
        for i, (op_name, (op_norm, op_size)) in enumerate(op_norm_normal_dict.items()):
            op_names.append(op_name)
            op_norm_normalized = op_norm / (op_size ** normalization_exponent)
            plt.semilogy(i, op_norm_normalized.item(), "o")
        
    plt.xticks(np.arange(num_ops), op_names, rotation=90)
    plt.xlim(-1, num_ops)
    plt.ylim(1e-5, 1e5)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("{}_normal.png".format(filename))
    plt.close()
    
    
    num_ops = len(op_norm_reduce_dict)
    f2 = plt.figure(num=None, figsize=(num_ops*0.15, 6), dpi=100, facecolor='w', edgecolor='k')    
    op_names = []
    if normalization == "none":
        for i, (op_name, (op_norm, _)) in enumerate(op_norm_reduce_dict.items()):
            op_names.append(op_name)
            plt.semilogy(i, op_norm.item(), "o")
    elif normalization == "mul":
        for i, (op_name, (op_norm, op_size)) in enumerate(op_norm_reduce_dict.items()):
            op_names.append(op_name)
            op_norm_normalized = op_norm * (op_size ** normalization_exponent)
            plt.semilogy(i, op_norm_normalized.item(), "o")
    elif normalization == "div":
        for i, (op_name, (op_norm, op_size)) in enumerate(op_reduce_normal_dict.items()):
            op_names.append(op_name)
            op_norm_normalized = op_norm / (op_size ** normalization_exponent)
            plt.semilogy(i, op_norm_normalized.item(), "o")
        
    plt.xticks(np.arange(num_ops), op_names, rotation=90)
    plt.xlim(-1, num_ops)
    plt.ylim(1e-5, 1e5)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("{}_reduce.png".format(filename))
    plt.close()
    


def discretize_search_model_by_cell(model_path, network_eval, network_search, threshold, CIFAR_CLASSES = 10, normalization="none", normalization_exponent=0):
  """
  remove the ops with a small norm and the discrete cell will be scaled up for evaluation
  
  All suboperations of an operation are grouped into a single vector. For example, op 3 of edge 7 has the following suboperations: _ops.7._ops.3.op.1.weight, _ops.7._ops.3.op.2.weight, _ops.7._ops.3.op.5.weight, _ops.7._ops.3.op.6.weight. All of these suboperations are grouped into a single vector called "_ops.7._ops.3". During discretization, the operation "_ops.7._ops.3" will be pruned if its norm is smaller than the pruning threshold.
  """
  model = SearchNetwork(network_search.init_channels, CIFAR_CLASSES, network_search.cells, network_search.criterion, network_search.ops)
  model = model.cuda()
  model.load_state_dict(torch.load(model_path))

  model_params = group_model_params_by_cell(model, network_search)
  op_norm_normal, op_norm_reduce = compute_op_norm_across_cells(model_params)

  alpha_normal = []
  alpha_edge = []
  edge_index = 0  
  for op_index, (op_name, (op_norm, op_size)) in enumerate(op_norm_normal.items()): # iterate over the operations (not suboperations)
        if  edge_index * network_search.num_ops <= op_index < (edge_index + 1) * network_search.num_ops:
            if normalization == "none":
                op_norm_normalized = op_norm
            elif normalization == "mul":
                op_norm_normalized = op_norm * (op_size ** normalization_exponent)
            elif normalization == "div":
                op_norm_normalized = op_norm / (op_size ** normalization_exponent)
                
            if op_norm_normalized <= threshold:
                alpha_edge.append(0)
            else:
                alpha_edge.append(1)
            if op_index == (edge_index + 1) * network_search.num_ops - 1:
                alpha_normal.append(alpha_edge)
                alpha_edge = []
                edge_index += 1
  alpha_normal = torch.tensor(alpha_normal)

  alpha_reduce = []
  alpha_edge = []
  edge_index = 0
  for op_index, (op_name, (op_norm, op_size)) in enumerate(op_norm_reduce.items()):        
        if  edge_index * network_search.num_ops <= op_index < (edge_index + 1) * network_search.num_ops:
            if normalization == "none":
                op_norm_normalized = op_norm
            elif normalization == "mul":
                op_norm_normalized = op_norm * (op_size ** normalization_exponent)
            elif normalization == "div":
                op_norm_normalized = op_norm / (op_size ** normalization_exponent)
                
            if op_norm_normalized <= threshold:
                alpha_edge.append(0)
            else:
                alpha_edge.append(1)
            if op_index == (edge_index + 1) * network_search.num_ops - 1:
                alpha_reduce.append(alpha_edge)
                alpha_edge = []
                edge_index += 1
  alpha_reduce = torch.tensor(alpha_reduce)

  alpha_network = []
  num_reduce_cell = len(network_eval.reduce_cell_indices)
  cur_reduce_cell = 0
  for cell_index in range(network_eval.cells): # cells up to the last reduce cell (included)
    if cell_index < network_eval.reduce_cell_indices[cur_reduce_cell]:
        alpha_network.append((False, np.vstack(alpha_normal)))
    elif cell_index == network_eval.reduce_cell_indices[cur_reduce_cell]:
        alpha_network.append((True,  np.vstack(alpha_reduce)))
        cur_reduce_cell += 1        
        if cur_reduce_cell == num_reduce_cell:
            break            
  # cells after the last reduce cell            
  for cell_index in range(network_eval.reduce_cell_indices[-1]+1, network_eval.cells):
        alpha_network.append((False, np.vstack(alpha_normal)))
        
  genotype_network = get_genotype(model.genotype(), alpha_network)

  return alpha_network, genotype_network



        
def get_genotype(genotype_supernet, alpha_network):
    genotype_network = []
    for i, (reduce_cell, alpha_cell) in enumerate(alpha_network):
        alpha_cell = alpha_cell.flatten()
        indices = np.where(alpha_cell == 1)[0]
        if reduce_cell:
            genotype_network.append([genotype_supernet.reduce[x] for x in indices.astype(int)])
        else:
            genotype_network.append([genotype_supernet.normal[x] for x in indices.astype(int)])
    return genotype_network

def visualize_cell(alpha, steps, primitives, filename):
  colors = ["sienna3", "red", "green4", "royalblue", "magenta"]
  num_colors = len(colors)
  g = Digraph(
      format="pdf",
      edge_attr=dict(fontsize='10'),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2'),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  # input node
  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  
  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  edge_offset = 0
  active_nodes = []
  color_index = 0
  for step_index in range(2, steps + 2):
    for edge_index in range(edge_offset, edge_offset + step_index):
      primitives_edge = primitives[edge_index]
      if sum(alpha[edge_index]) == 0:
        continue
      
      if step_index-2 not in active_nodes:
        active_nodes.append(step_index-2)
      for op_index, active_op in enumerate(alpha[edge_index]):
        if active_op:
            op = primitives_edge[op_index]
            j = edge_index - edge_offset
            
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(step_index - 2)
            color = colors[color_index]
            g.edge(u, v, label=op, color=color, fillcolor=color, fontcolor=color)
            color_index = (color_index+1) % num_colors
    edge_offset += step_index

  """output node"""
  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in active_nodes:
    g.edge(str(i), "c_{k}", fillcolor="gray", penwidth="2.0")

  g.render(filename, view=False)
  os.remove(filename)

    
def visualize_cell_in_network(network, alpha_network, scale_type, folder_path):
  if scale_type == "cell":
    visualize_cell(alpha_network[network.reduce_cell_indices[0]-1][1], network.steps, network.ops['primitives_normal'], "{}/cell_normal".format(folder_path))
    visualize_cell(alpha_network[network.reduce_cell_indices[0]][1], network.steps, network.ops['primitives_reduct'], "{}/cell_reduce".format(folder_path))
    
    
def discretize_model_by_operation(model, network_eval, genotype, threshold, folder_path, num_edges = 8):       
    edge_offset = [0] #edge_offset = [0, 2, 5, 9]
    for i in range(2, network_eval.steps + 2 - 1):
        edge_offset.append(edge_offset[-1] + i)
        
    alpha_cell_list = []
    alpha_network = []
    cell_inactive = []
    for cell_index, m in enumerate(model.cells):
        assert cell_index <= network_eval.cells - 1, "The number of cells in the loaded model is different from the number of cells expected ({}).".format(network_eval.cells)
        
        alpha_cell = np.zeros((network_eval.num_edges, network_eval.num_ops))
        if cell_index in network_eval.reduce_cell_indices:
            reduce_cell = True
            op_names, indices = zip(*genotype.reduce)
        else:
            reduce_cell = False
            op_names, indices = zip(*genotype.normal)

        for edge_index, (op_name, index) in enumerate(zip(op_names, indices)):
            op_index = network_eval.ops.index(op_name)
            node_index = edge_index // 2
            alpha_cell[edge_offset[node_index] + index][op_index] = 1
            
        for name, param in m.named_parameters():
            if "_ops" in name:
                if "bias" in name:
                    continue
                edge_index = int(name[5])
                node_index = edge_index // 2
                op_name = op_names[edge_index]
                index = indices[edge_index]
                op_index = network_eval.ops.index(op_name)
                alpha_cell[edge_offset[node_index] + index][op_index] *= (torch.norm(param) > threshold)
        alpha_cell_list.append(alpha_cell)
        cell_inactive.append(np.sum(alpha_cell) == 0)
        
#         print("cell {}, alpha_cell {}".format(cell_index, alpha_cell))
        
    assert cell_index == network_eval.cells - 1, "The number of cells in the loaded model is different from the number of cells expected ({}).".format(network_eval.cells)

    """detecting redundant edges..."""
#     print("detecting redundant edges...")
    
    genotype_network = []
    for cell_index in range(0, network_eval.cells):
        
        alpha_cell = alpha_cell_list[cell_index]
        if cell_index == 0:
            node_inactive_list = [False, False]
        elif cell_index == 1:
            node_inactive_list = [False, cell_inactive[cell_index - 1]]
        else:
            node_inactive_list = [cell_inactive[cell_index - 2], cell_inactive[cell_index - 1]]
        for node_index in range(2, network_eval.steps + 2):
            for edge_index in range(edge_offset[node_index-2], edge_offset[node_index-2] + node_index):
#                 print("    edge_index {}".format(edge_index))
                if node_inactive_list[edge_index - edge_offset[node_index-2]]:
                    for op_index in range(network_eval.num_ops):
                        alpha_cell[edge_index][op_index] = 0
               
            num_active_ops = 0
            for edge_index in range(edge_offset[node_index-2], edge_offset[node_index-2] + node_index):
                for op_index in range(network_eval.num_ops):
                    num_active_ops += alpha_cell[edge_index][op_index]
                        
            step_inactive = (num_active_ops == 0)
#             print("node_index {}, num_active_ops: {}".format(node_index - 2, num_active_ops))
            
            node_inactive_list.append(step_inactive)
        cell_inactive[cell_index] = (sum(node_inactive_list[2:]) == network_eval.steps)
        alpha_network.append((cell_index in network_eval.reduce_cell_indices, alpha_cell))
                
#         print("cell {}, alpha_cell {}".format(cell_index, alpha_cell))
#         print("inactive node list: {}".format(node_inactive_list))
#         print("cell inactive? {}".format(cell_inactive[cell_index]))
        
        if cell_index in network_eval.reduce_cell_indices:
            primitives_cell = network_eval.ops['primitives_reduct']
        else:
            primitives_cell = network_eval.ops['primitives_normal']
        visualize_cell(alpha_cell, network.steps, primitives_cell, "{}/cell_{:02d}".format(folder_path, cell_index))

        genotype_cell = []
        for node_index in range(network_eval.steps):
            for edge_index in range(edge_offset[node_index], edge_offset[node_index] + node_index + 2):
                for kk in range(network_eval.num_ops):
                    if alpha_cell[edge_index][kk] == 1:
                        op_name = network_eval.ops[kk]
                        source_node = edge_index - edge_offset[node_index]
                        genotype_cell.append((op_name, source_node))
#         print(genotype_cell)
    
        genotype_network.append(genotype_cell)
        
    return alpha_network, genotype_network