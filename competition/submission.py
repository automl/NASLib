from naslib.search_spaces.core.graph import Graph
from naslib.predictors.predictor import Predictor
from compute import count_parameters

class ZeroCostPredictor(Predictor):
    """ A sample submission class.

    CodaLab will import this class from your submission.py, and evaluate it.
    Your class must be named `ZeroCostPredictor`, and must contain self.method_type
    with your method name in it.
    """
    def __init__(self):
        self.method_type = 'MyZeroCostPredictorName'

    def pre_process(self) -> None:
        """ This method is called exactly once before query is called repeatedly with the models to score """
        pass

    def query(self, graph: Graph, dataloader=None) -> float:
        """ Predict the score of the given model

        Args:
            graph       : Model to score
            dataloader  : DataLoader for the task to predict. E.g., if the task is to
                          predict the score of a model for classification on CIFAR10 dataset,
                          a CIFAR10 Dataloader is passed here.

        Returns:
            Score of the model. Higher the score, higher the model is ranked.
        """

        # You can consume the dataloader and pass the data through the model here
        # data, labels = next(iter(dataloader))
        # logits = graph(data)

        # In this example, however, we simply count the number of parameters in the model
        # This zero-cost predictor thus gives higher score to larger models.
        score = count_parameters(graph)

        # Higher the score, higher the ranking of the model
        return score
