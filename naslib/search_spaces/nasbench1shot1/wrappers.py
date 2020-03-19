from collections import namedtuple
from nasbench import api


Architecture = namedtuple('Architecture', ['adjacency_matrix', 'node_list'])


class Model(object):
    """A class representing a model.

    It holds two attributes: `arch` (the simulated architecture) and `accuracy`
    (the simulated accuracy / fitness). See Appendix C for an introduction to
    this toy problem.

    In the real case of neural networks, `arch` would instead hold the
    architecture of the normal and reduction cells of a neural network and
    accuracy would be instead the result of training the neural net and
    evaluating it on the validation set.

    We do not include test accuracies here as they are not used by the algorithm
    in any way. In the case of real neural networks, the test accuracy is only
    used for the purpose of reporting / plotting final results.

    In the context of evolutionary algorithms, a model is often referred to as
    an "individual".

    Attributes:  (as in the original code)
      arch: the architecture as an int representing a bit-string of length `DIM`.
          As a result, the integers are required to be less than `2**DIM`. They
          can be visualized as strings of 0s and 1s by calling `print(model)`,
          where `model` is an instance of this class.
      accuracy:  the simulated validation accuracy. This is the sum of the
          bits in the bit-string, divided by DIM to produce a value in the
          interval [0.0, 1.0]. After that, a small amount of Gaussian noise is
          added with mean 0.0 and standard deviation `NOISE_STDEV`. The resulting
          number is clipped to within [0.0, 1.0] to produce the final validation
          accuracy of the model. A given model will have a fixed validation
          accuracy but two models that have the same architecture will generally
          have different validation accuracies due to this noise. In the context
          of evolutionary algorithms, this is often known as the "fitness".
    """

    def __init__(self):
        self.arch = None
        self.validation_accuracy = None
        self.test_accuracy = None
        self.training_time = None
        self.budget = None

    def update_data(self, arch, nasbench_data, budget):
        self.arch = arch
        self.validation_accuracy = nasbench_data['validation_accuracy']
        self.test_accuracy = nasbench_data['test_accuracy']
        self.training_time = nasbench_data['training_time']
        self.budget = budget

    def query_nasbench(self, nasbench, sample):
        config = ConfigSpace.Configuration(
            search_space.get_configuration_space(), vector=sample
        )
        adjacency_matrix, node_list = search_space.convert_config_to_nasbench_format(config)
        if type(search_space) == SearchSpace3:
            node_list = [INPUT, *node_list, OUTPUT]
        else:
            node_list = [INPUT, *node_list, CONV1X1, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)

        nasbench_data = nasbench.query(model_spec)
        self.arch = Architecture(adjacency_matrix=adjacency_matrix,
                                 node_list=node_list)
        self.validation_accuracy = nasbench_data['validation_accuracy']
        self.test_accuracy = nasbench_data['test_accuracy']
        self.training_time = nasbench_data['training_time']


class NasbenchWrapper(api.NASBench):
    """Small modification to the NASBench class, to return all three architecture evaluations at
    the same time, instead of samples."""

    def query(self, model_spec, epochs=108, stop_halfway=False):
        """Fetch one of the evaluations for this model spec.

        Each call will sample one of the config['num_repeats'] evaluations of the
        model. This means that repeated queries of the same model (or isomorphic
        models) may return identical metrics.

        This function will increment the budget counters for benchmarking purposes.
        See self.training_time_spent, and self.total_epochs_spent.

        This function also allows querying the evaluation metrics at the halfway
        point of training using stop_halfway. Using this option will increment the
        budget counters only up to the halfway point.

        Args:
          model_spec: ModelSpec object.
          epochs: number of epochs trained. Must be one of the evaluated number of
            epochs, [4, 12, 36, 108] for the full dataset.
          stop_halfway: if True, returned dict will only contain the training time
            and accuracies at the halfway point of training (num_epochs/2).
            Otherwise, returns the time and accuracies at the end of training
            (num_epochs).

        Returns:
          dict containing the evaluated darts for this object.

        Raises:
          OutOfDomainError: if model_spec or num_epochs is outside the search space.
        """
        if epochs not in self.valid_epochs:
            raise api.OutOfDomainError('invalid number of epochs, must be one of %s'
                                       % self.valid_epochs)

        fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
        trainings = []
        for index in range(self.config['num_repeats']):
            computed_stat_at_epoch = computed_stat[epochs][index]

            data = {}
            data['module_adjacency'] = fixed_stat['module_adjacency']
            data['module_operations'] = fixed_stat['module_operations']
            data['trainable_parameters'] = fixed_stat['trainable_parameters']

            if stop_halfway:
                data['training_time'] = computed_stat_at_epoch['halfway_training_time']
                data['train_accuracy'] = computed_stat_at_epoch['halfway_train_accuracy']
                data['validation_accuracy'] = computed_stat_at_epoch['halfway_validation_accuracy']
                data['test_accuracy'] = computed_stat_at_epoch['halfway_test_accuracy']
            else:
                data['training_time'] = computed_stat_at_epoch['final_training_time']
                data['train_accuracy'] = computed_stat_at_epoch['final_train_accuracy']
                data['validation_accuracy'] = computed_stat_at_epoch['final_validation_accuracy']
                data['test_accuracy'] = computed_stat_at_epoch['final_test_accuracy']

            self.training_time_spent += data['training_time']
            if stop_halfway:
                self.total_epochs_spent += epochs // 2
            else:
                self.total_epochs_spent += epochs
            trainings.append(data)

        return trainings

