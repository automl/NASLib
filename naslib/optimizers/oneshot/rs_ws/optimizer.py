import torch
import logging

from naslib.optimizers import OneShotNASOptimizer
from naslib.search_spaces.nasbench301.conversions import convert_compact_to_genotype

logger = logging.getLogger(__name__)


class RandomNASOptimizer(OneShotNASOptimizer):
    """
    Implementation of the Random NAS with weight sharing as in
        Li et al. 2019: Random Search and Reproducibility for Neural Architecture Search.
    """

    @staticmethod
    def add_alphas(edge):
        """
        Function to add the architectural weights to the edges.
        """
        len_primitives = len(edge.data.op)
        alpha = torch.nn.Parameter(
            torch.zeros(size=[len_primitives], requires_grad=False)
        )
        edge.data.set("alpha", alpha, shared=True)

    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val

        # Update architecture weights by sampling only a random arch and
        # setting the alpha values accordingly
        self.sample_random_and_update_alphas()

        logits_val = self.graph(input_val)
        val_loss = self.loss(logits_val, target_val)
        val_loss.backward()

        # Update op weights
        self.op_optimizer.zero_grad()
        logits_train = self.graph(input_train)
        train_loss = self.loss(logits_train, target_train)
        train_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
        self.op_optimizer.step()

        return logits_train, logits_val, train_loss, val_loss

    def sample_random_and_update_alphas(self):
        tmp_graph = self.search_space.clone()
        tmp_graph.sample_random_architecture()

        if self.graph.get_type() == "nasbench201":
            sample = tmp_graph.get_op_indices()
        elif self.graph.get_type() == "nasbench301":
            sample = convert_compact_to_genotype(tmp_graph.get_compact())

        self.set_alphas_from_path(sample)
