#####################################################
################# START NEW CELL ####################

from naslib.predictors import ZeroCost
from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_train_val_loaders, get_project_root
from fvcore.common.config import CfgNode
from tqdm import tqdm

# Create configs required for get_train_val_loaders
config = {
    'dataset': 'cifar10', # Dataset to loader: can be cifar100, svhn, ImageNet16-120, jigsaw, class_object, class_scene, or autoencoder (the last four are TNB101 datasets)
    'data': str(get_project_root()) + '/data', # path to naslib/data
    'search': {
        'seed': 9001, # Seed to use in the train, validation and test dataloaders
        'train_portion': 0.7, # Portion of train dataset to use as train dataset. The rest is used as validation dataset.
        'batch_size': 32, # batch size of the dataloaders
    }
}
config = CfgNode(config)

# Get the dataloaders
train_loader, val_loader, test_loader, train_transform, valid_transform = get_train_val_loaders(config)

# Sample a random NB201 graph and instantiate it
graph = NasBench201SearchSpace()
graph.sample_random_architecture()
graph.parse()

# Instantiate the ZeroCost predictor
# The Zero Cost predictors can be any of the following:
# {'epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'plain', 'snip', 'synflow', 'zen', 'flops', 'params'}

zc_pred = 'l2_norm'
zc_predictor = ZeroCost(method_type=zc_pred)
score = zc_predictor.query(graph=graph, dataloader=train_loader)

print(f'Score of model for Zero Cost predictor {zc_pred}: {score}')

#####################################################
################# START NEW CELL ####################

# Instantiate the ZeroCost predictor
# The Zero Cost predictors can be any of the following:
zc_predictors = ['epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'plain', 'snip', 'synflow', 'zen', 'flops', 'params']

print('Scores of model')
for zc_pred in zc_predictors:
    zc_predictor = ZeroCost(method_type=zc_pred)
    score = zc_predictor.query(graph=graph, dataloader=train_loader)
    print(f'{zc_pred}: {score}')

#####################################################
################# START NEW CELL ####################

from naslib.search_spaces.core import Metric
from naslib.utils import compute_scores, get_dataset_api

# Sample 50 random architectures, query their performances
n_graphs = 50
models = []
val_accs = []
zc_scores = []

print('Loading NAS-Bench-201 API...')
api = get_dataset_api('nasbench201', 'cifar10')

print(f'Sampling {n_graphs} NAS-Bench-201 models...')
for _ in tqdm(range(n_graphs)):
    graph = NasBench201SearchSpace()
    graph.sample_random_architecture()
    graph.parse()

    models.append(graph)

print('Querying validation performance for all models')
for graph in tqdm(models):
    acc = graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10', dataset_api=api)
    val_accs.append(acc)

zc_predictor = ZeroCost(method_type='jacov')

print('Scoring the models using Zero Cost predictor (jacov)')
for graph in tqdm(models):
    score = zc_predictor.query(graph, dataloader=train_loader)
    zc_scores.append(score)

correlations = compute_scores(ytest=val_accs, test_pred=zc_scores)
kendalltau_corr = correlations['kendalltau']
spearman_corr = correlations['spearman']
pearson_corr = correlations['pearson']

print('*'*50)
print('Validation accuracies: ', val_accs)
print()
print('Zero Cost predictor scores: ', zc_scores)
print('*'*50)
print('Correlations between validation accuracies (ground truth) and Zero Cost predictor scores (prediction): ')
print('Kendall Tau correlation:', kendalltau_corr)
print('Spearman correlation:', spearman_corr)
print('Pearson correlation:', pearson_corr)

#####################################################
################# START NEW CELL ####################

from naslib.utils import get_zc_benchmark_api

zc_api = get_zc_benchmark_api('nasbench201', 'cifar10')
graph = models[0]

# Use the Zero Cost Benchmark to get the score for the model for a particular ZC proxy
pred = 'grasp'
spec = graph.get_hash()
score = zc_api[str(spec)][pred]['score']
time_to_compute = zc_api[str(spec)][pred]['time']

print(f'Score of model with spec {spec} for Zero Cost predictor {pred}: {score}')
print(f'Time taken to compute the score for the model: {time_to_compute:.2f}s')

#####################################################
################# START NEW CELL ####################

zc_scores = {pred: [] for pred in zc_predictors} # Just dictionary of an empty list for each predictor
print('zc_scores:', zc_scores)

print('Querying Zero Cost Benchmark for scores')
for graph in tqdm(models):
    spec = graph.get_hash()

    # Get the score for this model for all the Zero Cost predictors
    for pred in zc_predictors:
        score = zc_api[str(spec)][pred]['score']
        zc_scores[pred].append (score)

# Print the Zero Cost values for the first 5 models
for pred in zc_predictors:
    print(pred, zc_scores[pred][:5])

kt_corrs = {}
# Compute the rank correlation for each of the predictors
for pred in zc_predictors:
    correlations = compute_scores(ytest=val_accs, test_pred=zc_scores[pred])
    kendalltau_corr = correlations['kendalltau']
    kt_corrs[pred] = kendalltau_corr

print('Kendall-Tau correlations for all the predictors:')
print(kt_corrs)

# Sort the predictors from best to worst for these models
kt_corrs_list = [(pred, score) for pred, score in kt_corrs.items()]
kt_corrs_list = sorted(kt_corrs_list, reverse=True, key=lambda pred_score_tuple: pred_score_tuple[1])

print('ZC predictors ranked (best to worst):')
for idx, (pred, score) in enumerate(kt_corrs_list):
    print(f'#{idx+1}', pred, score)