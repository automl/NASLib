from naslib.search_spaces.core.graph import Graph
from torch.utils.data import DataLoader
class Predictor:
    def __init__(self, ss_type=None):
        self.ss_type = ss_type

    def set_ss_type(self, ss_type):
        self.ss_type = ss_type

    def pre_process(self):
        """
        This is called at the beginning of the NAS algorithm, before any
        architectures have been queried. Use it for any kind of one-time set up.
        """
        pass

    def fit(self, xtrain, ytrain) -> None:
        """
        This can be called any number of times during the NAS algorithm.
        input: list of architectures, list of architecture accuracies
        output: none
        """
        pass

    def query(self, graph: Graph, dataloader: DataLoader) -> float:
        """ Predict the score of the given model

        Args:
            graph       : Model to score
            dataloader  : DataLoader for the task to predict. E.g., if the task is to
                          predict the score of a model for classification on CIFAR10 dataset,
                          a CIFAR10 Dataloader is passed here.

        Returns:
            Score of the model. Higher the score, higher the model is ranked.
        """
        pass
