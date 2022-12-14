### Changes

- Modified the initializers of the following search spaces:
    1. `NasBench101SearchSpace(stacks=3, channels=64)` to `NasBench101SearchSpace(n_classes=10)`
    2. `NasBench201SearchSpace()` to `NasBench201SearchSpace(n_classes=10, in_channels=3)`
    3. `DartsSearchSpace()` to `NasBench301SearchSpace(n_classes=10, in_channels=3, auxillery=True)`
    4. `TransBench101SearchSpaceMicro(dataset='jigsaw')` to `TransBench101SearchSpaceMicro(dataset='jigsaw', use_small_model=True, create_graph=False, n_classes=10, in_channels=3)`
    5. `TransBench101SearchSpaceMacro()` to `TransBench101SearchSpaceMacro(dataset='jigsaw')`

- Modified the initializers of the following primitive components:
    1. `Stem(C_out)` to `Stem(C_in=3, C_out=64)`
    2. `StemJigsaw(C_out)` to `StemJigsaw(C_in=3, C_out=64)`

- Modified the initializer of:
    - `RandomSearch(config, weight_optimizer=torch.optim.SGD, loss_criteria=torch.nn.CrossEntropyLoss(), grad_clip=None)` to `RandomSearch(config)`
- Introduced the boolean parameter `load_labeled`, which determines whether to sample from a list of labeled architectures[^1] or from a benchmark, to the following search spaces:
    1. `NasBench101SearchSpace`
    2. `NasBench201SearchSpace`
    3. `NasBench301SearchSpace`
    4. `TransBench101SearchSpaceMicro`
    5. `TransBench101SearchSpaceMacro`
- Introduced the parameter `instantiate_model`[^2], which determines whether a graph model is created, to the following search spaces:
    1. `NasBench101SearchSpace`
    2. `NasBench201SearchSpace`
    3. `NasBench301SearchSpace`
    4. `TransBench101SearchSpaceMicro`
    5. `TransBench101SearchSpaceMacro`

[^1]: `search_space.labeled_archs` should be set to a list of architecture specs upon initialization whenever `load_labeled=True`
[^2]: `search_space.instantiate_model` should be set to `False` only when a benchmark is being used with `acq_fn_optimization="random_sampling"`, otherwise set `search_space.instantiate_model=True` and call `arch.parse()` whenever a new architecture is sampled. When `search_space.instantiate_model=False` the sampled architectures are not live models and the function `search_space.parse()` cannot be called.
