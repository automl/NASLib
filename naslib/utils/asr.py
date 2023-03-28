import re
import pickle
import random
import pathlib
import functools
import copy
import hashlib
import numpy as np
import collections.abc as cabc


def get_model_graph(arch_vec, ops=None, minimize=True, keep_dims=False):
    if ops is None:
        import search_space as ss # TODO: Fix this
        ops = ss.all_ops
    num_nodes = len(arch_vec)
    mat = np.zeros((num_nodes+2, num_nodes+2))
    labels = ['input']
    prev_skips = []
    for nidx, node in enumerate(arch_vec):
        op = node[0]
        labels.append(ops[op])
        mat[nidx, nidx+1] = 1
        for i, sc in enumerate(prev_skips):
            if sc:
                mat[i, nidx+1] = 1
        prev_skips = node[1:]
    labels.append('output')
    mat[num_nodes, num_nodes+1] = 1
    for i, sc in enumerate(prev_skips):
        if sc:
            mat[i, num_nodes+1] = 1
    orig = None
    if minimize:
        orig = copy.copy(mat), copy.copy(labels)
        for n in range(len(mat)):
            if labels[n] == 'zero':
                for n2 in range(len(mat)):
                    if mat[n,n2]:
                        mat[n,n2] = 0
                    if mat[n2,n]:
                        mat[n2,n] = 0
        def bfs(src, mat, backward):
            visited = np.zeros(len(mat))
            q = [src]
            visited[src] = 1
            while q:
                n = q.pop()
                for n2 in range(len(mat)):
                    if visited[n2]:
                        continue
                    if (backward and mat[n2,n]) or (not backward and mat[n,n2]):
                        q.append(n2)
                        visited[n2] = 1
            return visited
        vfw = bfs(0, mat, False)
        vbw = bfs(len(mat)-1, mat, True)
        v = vfw + vbw
        dangling = (v < 2).nonzero()[0]
        if dangling.size:
            if keep_dims:
                mat[dangling, :] = 0
                mat[:, dangling] = 0
                for i in dangling:
                    labels[i] = None
            else:
                mat = np.delete(mat, dangling, axis=0)
                mat = np.delete(mat, dangling, axis=1)
                for i in sorted(dangling, reverse=True):
                    del labels[i]
    return (mat, labels), orig


def graph_hash(g):
    all_ops = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'zero']
    m, l = g
    def hash_module(matrix, labelling):
        """Computes a graph-invariance MD5 hash of the matrix and label pair.
        Args:
            matrix: np.ndarray square upper-triangular adjacency matrix.
            labelling: list of int labels of length equal to both dimensions of
                matrix.
        Returns:
            MD5 hash of the matrix and labelling.
        """
        vertices = np.shape(matrix)[0]
        in_edges = np.sum(matrix, axis=0).tolist()
        out_edges = np.sum(matrix, axis=1).tolist()
        assert len(in_edges) == len(out_edges) == len(labelling), f'{labelling} {matrix}'
        hashes = list(zip(out_edges, in_edges, labelling))
        hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
        # Computing this up to the diameter is probably sufficient but since the
        # operation is fast, it is okay to repeat more times.
        for _ in range(vertices):
            new_hashes = []
            for v in range(vertices):
                in_neighbours = [hashes[w] for w in range(vertices) if matrix[w, v]]
                out_neighbours = [hashes[w] for w in range(vertices) if matrix[v, w]]
                new_hashes.append(hashlib.md5(
                        (''.join(sorted(in_neighbours)) + '|' +
                        ''.join(sorted(out_neighbours)) + '|' +
                        hashes[v]).encode('utf-8')).hexdigest())
            hashes = new_hashes
        fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()
        return fingerprint
    labels = []
    if l:
        labels = [-1] + [all_ops.index(op) for op in l[1:-1]] + [-2]
    return hash_module(m, labels)


def get_model_hash(arch_vec, ops=None, minimize=True):
    ''' Get hash of the architecture specified by arch_vec.
        Architecture hash can be used to determine if two
        configurations from the search space are in fact the
        same (graph isomorphism).
    '''
    g, _ = get_model_graph(arch_vec, ops=ops, minimize=minimize)
    return graph_hash(g)


class _Dataset():
    def __init__(self, dataset_files, validate_data, db_type):
        if isinstance(dataset_files, str):
            dataset_files = [dataset_files]

        self.dbs = []
        self.header = None
        if db_type == 'training':
            self.seeds = []
        elif db_type == 'benchmarking':
            self.devices = []
        elif db_type == 'static':
            if len(dataset_files) != 1:
                raise ValueError('Expected exactly one dataste file')

        for db_file in dataset_files:
            with open(db_file, 'rb') as f:
                header = pickle.load(f)
                if header['dataset_type'] != db_type:
                    raise ValueError(f'Expected a dataset file with {db_type} information')

                if db_type == 'training':
                    seed = header.pop('seed')
                elif db_type == 'benchmarking':
                    device = header.pop('device')

                if self.header is None:
                    self.header = header
                if self.header != header:
                    raise ValueError('Different dataset files contain data for different settings')

                # TODO: we could relax this if needed
                if db_type == 'training':
                    if header['columns'][:3] != ['model_hash', 'val_per', 'test_per']:
                        raise ValueError('In the current implementation we expect the dataset to contain information in order: model hash, val PER, test PER')
                elif db_type == 'benchmarking':
                    if header['columns'][:2] != ['model_hash', 'latency']:
                        raise ValueError('In the current implementation we expect the dataset to contain information in order: model hash, latency')
                elif db_type == 'static':
                    if header['columns'][:2] != ['model_hash', 'params']:
                        print(header['columns'])
                        raise ValueError('In the current implementation we expect the dataset to contain information in order: model hash, number of parameters')

                if db_type == 'training':
                    self.seeds.append(seed)
                elif db_type == 'benchmarking':
                    self.devices.append(device)
                data = pickle.load(f)
                data_dict = { model_hash: rest for model_hash, *rest in data }
                self.dbs.append(data_dict)

        if not self.dbs:
            raise ValueError('At least one dataset should be read')

        if validate_data and len(self.dbs) > 1:
            #if db_type == 'training':
            models = { model_hash: model_pt for model_hash, (*_, model_pt) in self.dbs[0].items() }
            for fidx, db in enumerate(self.dbs[1:]):
                if len(db) != len(models):
                    raise ValueError(f'Dataset file at position {fidx+1} has {len(db)} entries but the one at position 0 has {len(models)}')
                for model_hash, (*_, model_pt) in db.items():
                    if model_hash not in models:
                        raise ValueError(f'{model_hash} is present in dataset file {fidx+1} but no in 0')
                    if db_type == 'training':
                        # even if this is not true, the same model hash should guarantee that the architectures are the same
                        # however, internally we'd expect the points to be the same
                        assert model_pt == models[model_hash]

    @property
    def version(self):
        ''' Version of the dataset.
        '''
        return self.header['version']

    @property
    def search_space(self):
        ''' Search space shape. A (potentially nested) list of integers identifying
            different choices and their related number of options.
        '''
        return self.header['search_space']['shape']

    @property
    def ops(self):
        ''' List of the operations which were considered when creating the dataset.
        '''
        return self.header['search_space']['ops']

    @property
    def nodes(self):
        ''' Number of nodes which was considered when creating the dataset.
        '''
        return self.header['search_space']['nodes']

    @property
    def columns(self):
        ''' Names of values stored in the dataset, in-order.
            Can be used to identify specific information from values returned by
            functions which do not convert their results to dictionaries.
            See the remaining API for more information.
        '''
        return self.header['columns']

    def __contains__(self, arch):
        h = get_model_hash(arch, ops=self.ops)
        return h in self.dbs[0]


class StaticInfoDataset(_Dataset):
    def __init__(self, dataset_file):
        super().__init__([dataset_file], False, 'static')

    def _get(self, model_hash, return_dict):
        r = self.dbs[0].get(model_hash)
        if return_dict and r is not None:
            return dict(zip(self.columns[1:], r))
        return r

    def params(self, arch):
        ''' Return the number of parameters in a specific architecture.

            Arguments:
                arch - a point from the search space identifying a model
                return_dict - (optional) determinates if the returned values will be provided
                    as a ``dict`` or a scalar value. A ``dict`` contains the same values as
                    the ``list`` but allows the user to extract them by their names, whereas
                    a list can be thought of as a single row in a table containing values only.
                    The user can map particular elements of the returned ``list`` by considering
                    the values in provided ``devices`` argument. Default: ``False``.

            Returns:
                ``None`` if information about a given ``arch`` cannot be found in the dataset,
                otherwise a ``dict`` or a ``list`` containing information about the model.
        '''
        model_hash = get_model_hash(arch, ops=self.ops)
        ret = self._get(model_hash, False)
        return ret[0]


class BenchmarkingDataset(_Dataset):
    ''' An object representing a queryable dataset containing benchmarking information
        of Nasbench-ASR models.

        The dataset is constructed by reading a set of pickle files containing
        information about models benchmarked on different devices.

        All the files used to create a single ``BenchmarkingDataset`` object should contian information
        about models coming from the same search space and can only differ by the type of device used.
        If you want to compare performance of models from different search spaces you'd need to create
        different objects for each case.
    '''
    def __init__(self, dataset_files, validate_data=True):
        ''' Create a new dataset by loading data from the provided list of files.

            If multiple files are given, they should contain information about models
            from the same search space, benchmarked on different devices.

            If ``validate_data`` is set to ``True``, the data from the files will be validated
            to check if it's consistent. If the files are known to be ok, the argument can be
            set to ``False`` to speed up loading time a little bit (or to hack the code if you know
            what you are doing).
        '''
        super().__init__(dataset_files, validate_data, 'benchmarking')

    def _get(self, model_hash, devices, ret_dict):
        if devices is None:
            devices = self.devices
            indices = list(range(len(self.devices)))
        else:
            if isinstance(devices, str):
                devices = [devices]
            indices = [self.devices.index(d) for d in devices]

        raw = [] if not ret_dict else {}
        for didx, device_name in zip(indices, devices):
            value = self.dbs[didx].get(model_hash)
            if value is None:
                return None
            if not ret_dict:
                raw.append(value)
            else:
                value = dict(zip(self.columns[1:], value))
                raw[device_name] = value

        return raw

    def latency(self, arch, devices=None, return_dict=False):
        ''' Return benchmarking information about a specific architecture on the provided
            devices from the dataset.

            Arguments:
                arch - a point from the search space identifying a model
                device - (optional) if provided, the returned will be information about
                    the model's performance when run on the device with the given name(s),
                    otherwise latency on all devices will be returned; accepted values are:
                    Str, List[Str] and None
                return_dict - (optional) determinates if the returned values will be provided
                    as a ``dict`` or a simple ``list``. A ``dict`` contains the same values as
                    the ``list`` but allows the user to extract them by their names, whereas
                    a list can be thought of as a single row in a table containing values only.
                    The user can map particular elements of the returned ``list`` by considering
                    the values in provided ``devices`` argument. Default: ``False``.

            Returns:
                ``None`` if information about a given ``arch`` cannot be found in the dataset,
                otherwise a ``dict`` or a ``list`` containing information about the model.

            Raises:
                ValueError - if invalid ``device`` is given 
        '''
        model_hash = get_model_hash(arch, ops=self.ops)
        return self._get(model_hash, devices, return_dict)


class Dataset(_Dataset):
    ''' An object representing a queryable NasBench-ASR dataset.

        The dataset is constructed by reading a set of pickle files containing training
        information about models using different configurations (different initialization
        seed and/or total number of epochs).

        The training information can be optionally extended with benchmarking and static
        (e.g. number of parameters) information.

        All the files used to create a single ``Dataset`` object should contian information
        about models trained in the same setting and can only differ by the initialization seed.
        If you want to compare performance of models in different settings, e.g. using full training
        or reduced training of 10 epochs, you'd need to create different objects for each case.
    '''
    def __init__(self, dataset_files,  devices_files=None, static_info=None, validate_data=True):
        ''' Create a new dataset by loading data from the provided list of files.

            If multiple files are given, they should contain information about models
            trained in the same setting, differing only by their initialization seed.

            If ``validate_data`` is set to ``True``, the data from the files will be validated
            to check if it's consistent. If the files are known to be ok, the argument can be
            set to ``False`` to speed up loading time a little bit (or to hack the code if you know
            what you are doing).
        '''
        super().__init__(dataset_files, validate_data, 'training')
        self.bench_info = None
        self.static_info = None
        if devices_files:
            self.bench_info = BenchmarkingDataset(devices_files, validate_data=validate_data)
        if static_info:
            self.static_info = StaticInfoDataset(static_info)

    @property
    def epochs(self):
        ''' Total number of epochs for which the models were trained when creating the dataset.
        '''
        return self.header['epochs']

    def _get_raw_info(self, seed_idx, model_hash):
        raw = self.dbs[seed_idx].get(model_hash)
        if raw is None:
            return None
        return [model_hash] + list(raw) + [self.seeds[seed_idx]]

    def _get_info_dict(self, seed_idx, model_hash):
        raw = self.dbs[seed_idx].get(model_hash)
        if raw is not None:
            raw = dict(zip(self.columns[1:], raw))
            raw[self.columns[0]] = model_hash
            raw['seed'] = self.seeds[seed_idx]
        return raw

    def _get_info(self, seed_idx, model_hash, return_dict):
        if return_dict:
            return self._get_info_dict(seed_idx, model_hash)
        else:
            return self._get_raw_info(seed_idx, model_hash)

    def _query(self, model_hash, seed, devices, include_static_info, return_dict):
        if seed is None:
            seed_idx = random.randrange(len(self.seeds))
        else:
            seed_idx = self.seeds.index(seed)

        ret = self._get_info(seed_idx, model_hash, return_dict)
        if devices != False and (devices is not None or self.bench_info):
            if not self.bench_info:
                raise ValueError('No benchmarking information attached')
            lat = self.bench_info._get(model_hash, devices, return_dict)
            if return_dict:
               ret.update(lat)
            else:
                ret.extend(lat)

        if include_static_info:
            if not self.static_info:
                raise ValueError('No static information attached')
            info = self.static_info._get(model_hash, return_dict)
            if return_dict:
                ret['info'] = info
            else:
                ret.append(info)

        return ret

    def full_info(self, arch, seed=None, devices=None, include_static_info=None, return_dict=True):
        ''' Return all information about a specific architecture from the dataset.
            If multiple seeds are available, the can either return information about
            a specific one or a random one.

            Arguments:
                arch - a point from the search space identifying a model
                seed - (optional) if provided, the returned will be information about
                    the model's performance when initialized with this particular seed,
                    otherwise information related to a randomly chosen seed from the list
                    if available ones will be used. Default: random seed
                devices - (optional) add information about benchmarking on the provided devices,
                    if ``None`` all available devices are included, otherwise should be a name of
                    the device or a list of names, can also be exactly ``False`` to avoid including
                    benchmarking information even when they are available
                include_static_info - (optional) include static information about the model,
                    such as number of parameters, if set to ``None`` static information will be
                    added only if available
                return_dict - (optional) determinates if the returned values will be provided
                    as a ``dict`` or a simple ``list``. A ``dict`` contains the same values as
                    the ``list`` but alolws the user to extract them by their names, whereas
                    a list can be thought of as a single row in a table containing values only.
                    The user can map particular elements of the returned ``list`` by considering
                    the values in ``columns``. Default: ``True``.

            Returns:
                ``None`` if information about a given ``arch`` cannot be found in the dataset,
                otherwise a ``dict`` or a ``list`` containing information about the model.

            Raises:
                ValueError - if invalid ``seed`` is given 
        '''
        model_hash = get_model_hash(arch, ops=self.ops)
        return self._query(model_hash, seed, devices, include_static_info, return_dict)

    def full_info_by_graph(self, graph, seed=None, devices=None, include_static_info=None, return_dict=True):
        ''' Return all information about an architecture identified by the provided model
            graph.
            If multiple seeds are available, the can either return information about
            a specific one or a random one.

            Arguments:
                graph - a graph of a model from the search space, obtained by calling
                    ``nasbench_asr.graph_utils.get_model_graph(arch)``
                seed - (optional) if provided, the returned will be information about
                    the model's performance when initialized with this particular seed,
                    otherwise information related to a randomly chosen seed from the list
                    if available ones will be used. Default: random seed
                devices - (optional) add information about benchmarking on the provided devices,
                    if ``None`` all available devices are included, otherwise should be a name of
                    the device or a list of names, can also be exactly ``False`` to avoid including
                    benchmarking information even when they are available
                include_static_info - (optional) include static information about the model,
                    such as number of parameters, if set to ``None`` static information will be
                    added only if available
                return_dict - (optional) determinates if the returned values will be provided
                    as a ``dict`` or a simple ``list``. A ``dict`` contains the same values as
                    the ``list`` but allows the user to extract them by their names, whereas
                    a list can be thought of as a single row in a table containing values only.
                    The user can map particular elements of the returned ``list`` by considering
                    the values in ``columns``. Default: ``True``.

            Returns:
                ``None`` if information about a given ``arch`` cannot be found in the dataset,
                otherwise a ``dict`` or a ``list`` containing information about the model.

            Raises:
                ValueError - if invalid ``seed`` is given 
        '''
        model_hash = graph_hash(graph)
        return self._query(model_hash, seed, devices, include_static_info, return_dict)

    def test_acc(self, arch, seed=None):
        ''' Return test PER of a model.

            Test PER is currently defined as the test PER of the model at epoch
            with the lowest validation PER.

            Arguments:
                arch - a point from the search space identifying a model
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)

            Returns:
                ``None`` if the dataset does not contain information about a model ``arch``,
                otherwise a scalar ``float``.
        '''
        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, return_dict=False)
        if info is None:
            return None
        return info[2]

    def val_acc(self, arch, epoch=None, best=True, seed=None):
        ''' Return validation PER of a model.

            The returned PER can be either the best PER or the PER at the last epoch.
            The maximum number of epochs to consider can be controlled by ``epoch``.

            If ``vals`` is a list of validation PERs, the returned value can be
            defined as:

                epoch = epoch if epoch is not None else len(vals)
                return min(vals[:epoch]) if best else vals[epoch-1]

            Arguments:
                arch - a point from the search space identifying a model
                epoch - (optional) number of epochs to consider, if not provided
                    all epochs will be considered (default: ``None``)
                best - (optional) return best validation PER from epoch 1 to the
                    maximum considered epochs, otherwise return PER at the last
                    considered epoch (default: ``True``)
                seed - (optional) an initialization seed to use, if not provided information
                    will be queried for a random seed (default: ``None``)

        '''
        info = self.full_info(arch, seed=seed, devices=False, include_static_info=False, return_dict=False)
        if info is None:
            return None
        if epoch is None:
            epoch = len(info[1])
        if best:
            return min(info[1][:epoch])
        else:
            return info[1][epoch-1]

    @functools.wraps(BenchmarkingDataset.latency)
    def latency(self, *args, **kwargs):
        if not self.bench_info:
            raise ValueError('No benchmarking information attached')

        return self.bench_info.latency(*args, **kwargs)

    @functools.wraps(StaticInfoDataset.params)
    def params(self, *args, **kwargs):
        if not self.static_info:
            raise ValueError('No static information attached')

        return self.static_info.params(*args, **kwargs)



def from_folder(folder, max_epochs=None, seeds=None, devices=None, include_static_info=False, validate_data=True):
    ''' Create a ``Dataset`` object from files in a given directory.
        Arguments control what subset of the files will be used.

        Recognizable files should have names following the pattern::

            - nb-asr-e{max_epochs}-{seed}.pickle for training datasets
            - nb-asr-bench-{device}.pickle for benchmarking datasets
            - nb-asr-info.pickle for static information dataset

        Arguments:
            max_epochs - load dataset files related to accuracy of models
                when trained with at most ``max_epochs`` of training.
                The related files should have a 'e{max_epochs}' component
                in their name. If the argument is ``None``, load the dataset
                related to full training.
            seeds - if not provided the created dataset will use all available
                seeds (each file should hold information about one seed only).
                Otherwise it can be a single value or a list seeds to use.
                The function will not check if the file(s) for the provided seed(s)
                exist(s) and will fail silently (i.e., the resulting
                dataset simply won't include results for the provided seed)
            devices - (optional) add information about benchmarking on the provided devices,
                if ``None`` all available devices are included, otherwise should be a name of
                the device or a list of names, can also be exactly ``False`` to avoid including
                benchmarking information even when they are available
            include_static_info - (optional) include static information about the model,
                such as number of parameters
            validate_data - passed to ``Dataset`` constructor, if ``True`` the dataset
                will be validated to check consistency of the data. Can be set to ``False``
                to speed up loading if the data is known to be valid.

        Raises:
            ValueError - if ``folder`` is not a directory or does not exist
            ValueError - if any of the loaded dataset files contain 
    '''
    f = pathlib.Path(folder).expanduser()
    if not f.exists() or not f.is_dir():
        raise ValueError(f'{folder} is not a directory')

    if max_epochs is None:
        max_epochs = 40

    max_epochs = f'e{max_epochs}-'

    if seeds is not None:
        if isinstance(seeds, cabc.Sequence) and not isinstance(seeds, str):
            seeds = '(' + '|'.join(map(str, seeds)) + ')'
        else:
            seeds = str(seeds)
    else:
        seeds = '[0-9]+'

    if devices != False:
        if devices is not None:
            if isinstance(devices, cabc.Sequence) and not isinstance(devices, str):
                devices = '(' + '|'.join(map(str, devices)) + ')'
            else:
                devices = str(devices)
        else:
            devices = '[a-zA-Z0-9-]+'

    datasets = []
    bench_info = []
    static_info = None

    regex = re.compile(f'nb-asr-{max_epochs}{seeds}.pickle')
    regex2 = re.compile(f'nb-asr-bench-{devices}.pickle') if devices else None
    for ff in f.iterdir():
        if ff.is_file():
            if regex.fullmatch(ff.name):
                datasets.append(str(ff))
            if devices and regex2.fullmatch(ff.name):
                bench_info.append(str(ff))
            if include_static_info and ff.name == 'nb-asr-info.pickle':
                static_info = str(ff)


    return Dataset(datasets, bench_info, static_info, validate_data=validate_data)
