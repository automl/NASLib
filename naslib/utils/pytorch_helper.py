from typing import Optional, Callable, Iterator

import torch.nn as nn
import torch.optim as optim


def create_optimizer(
        opt: str,
        params: Iterator[nn.Parameter],
        lr: float = 0.025,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        # filter_bias_and_bn: bool = True,
        # layer_decay: Optional[float] = None,
        # param_group_fn: Optional[Callable] = None,
        **kwargs,
):
    """ Create an optimizer.
        This optimizer factory belings to the repository https://github.com/rwightman/pytorch-image-models/ (/timm/optim/optim_factory.py)
        at version v0.6.12

        Args:
            opt: name of optimizer to create
            params (Iterator[nn.Parameter]): model parameters to optimize
            lr: initial learning rate
            weight_decay: weight decay to apply in optimizer
            momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
            filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
            **kwargs: extra optimizer specific kwargs to pass through
        Returns:
            Optimizer
    """
    if isinstance(params, Iterator):
        parameters = params
    else:
        raise ValueError("Currently only model parameters are supported")

    opt_lower = opt.lower()
    opt_args = dict(weight_decay=weight_decay, **kwargs)
    opt_args.setdefault('lr', lr)

    # basic SGD & related
    if opt_lower == 'sgd':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)

    # adaptive
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adamax':
        optimizer = optim.Adamax(parameters, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adagrad':
        opt_args.setdefault('eps', 1e-8)
        optimizer = optim.Adagrad(parameters, **opt_args)
    elif opt_lower == 'asgd':
        optimizer = optim.ASGD(parameters, **opt_args)
    elif opt_lower == 'sparceadam':
        optimizer = optim.SparseAdam(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'lbfgs':
        optimizer = optim.LBFGS(parameters, **opt_args)

    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer


def create_criterion(
        crit: str,
        reduction: str = 'mean',
        **kwargs
):
    """ Builds a criterion based on specified parameters.
        Criterions from PyTorch v1.9.0 are supported

        Args:
            crit: name of the criterion
            reduction: reduction to apply to the output: ``none``, ``mean`` or ``sum``. Default: ``mean``.
            **kwargs: extra optimizer specific kwargs to pass through
        Returns:
            Criterion
        """
    crit_lower = crit.lower()
    crit_args = dict(reduction=reduction, **kwargs)

    if crit_lower == 'l1loss':
        criterion = nn.L1Loss(**crit_args)
    elif crit_lower == 'mseloss':
        criterion = nn.MSELoss(**crit_args)
    elif crit_lower == 'crossentropyloss':
        criterion = nn.CrossEntropyLoss(**crit_args)
    elif crit_lower == 'ctcloss':
        criterion = nn.CTCLoss(**crit_args)
    elif crit_lower == 'nllloss':
        criterion = nn.NLLLoss(**crit_args)
    elif crit_lower == 'poissonnllloss':
        criterion = nn.PoissonNLLLoss(**crit_args)
    elif crit_lower == 'gaussiannllloss':
        criterion = nn.GaussianNLLLoss(**crit_args)
    # elif crit_lower == 'nllloss2d':   # incompatible with torch 1.9.0
    #     criterion = nn.NLLLoss2d(**crit_args)
    elif crit_lower == 'kldivloss':
        criterion = nn.KLDivLoss(**crit_args)
    elif crit_lower == 'bceloss':
        criterion = nn.BCELoss(**crit_args)
    elif crit_lower == 'bcewithlogitsloss':
        criterion = nn.BCEWithLogitsLoss(**crit_args)
    elif crit_lower == 'marginrankingloss':
        criterion = nn.MarginRankingLoss(**crit_args)
    elif crit_lower == 'hingeembeddingloss':
        criterion = nn.HingeEmbeddingLoss(**crit_args)
    elif crit_lower == 'multilabelmarginloss':
        criterion = nn.MultiLabelMarginLoss(**crit_args)
    elif crit_lower == 'huberloss':
        criterion = nn.HuberLoss(**crit_args)
    elif crit_lower == 'smoothl1loss':
        criterion = nn.SmoothL1Loss(**crit_args)
    elif crit_lower == 'softmarginloss':
        criterion = nn.SoftMarginLoss(**crit_args)
    elif crit_lower == 'multilabelsoftmarginloss':
        criterion = nn.MultiLabelSoftMarginLoss(**crit_args)
    elif crit_lower == 'cosineembeddingloss':
        criterion = nn.CosineEmbeddingLoss(**crit_args)
    elif crit_lower == 'multimarginloss':
        criterion = nn.MultiMarginLoss(**crit_args)
    elif crit_lower == 'tripletmarginloss':
        criterion = nn.TripletMarginLoss(**crit_args)
    elif crit_lower == 'tripletmarginwithdistanceloss':
        criterion = nn.TripletMarginWithDistanceLoss(**crit_args)

    else:
        assert False and "Invalid criterion"
        raise ValueError

    return criterion
