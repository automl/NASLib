import unittest

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from naslib.tabular.embeddings.embedding import *


class EmbeddingTest(unittest.TestCase):

    def test_learned_entity_embedding(self):
        print("Learned entity embedding config space:", LearnedEntityEmbedding.get_config_space(categorical_features=[True, False]))

        test_config = {"categorical_features": [True, False],
                       "min_unique_values_for_embedding": 5,
                       "dimension_reduction_0": 0.5}

        # Create some test data, one-hot-encode it and pass encoder to the embedding
        X_test = np.array([["A", "B", "C", "A"],[1,2,3,1]], dtype=object).transpose()
        categorical_features = test_config["categorical_features"]

        ohe = OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore")
        encoder = ColumnTransformer(transformers=[("ohe", ohe, [i for i, f in enumerate(categorical_features) if f])], remainder="passthrough")
        X_test = encoder.fit_transform(X_test)

        encoder.categories_ = np.array([])
        encoder.categorical_features = categorical_features

        test_embedding = LearnedEntityEmbedding(test_config, 2, encoder)
        print("Test forward:", test_embedding.forward(torch.from_numpy(X_test.astype(np.float32))))

    def test_no_embedding(self):
        print("No embedding config space:", NoEmbedding.get_config_space())

        test_config = {"categorical_features": [True, False],
                       "min_unique_values_for_embedding": 5,
                       "dimension_reduction_0": 0.5}

        # Create some test data, one-hot-encode it and pass encoder to the embedding
        X_test = np.array([["A", "B", "C", "A"],[1,2,3,1]], dtype=object).transpose()
        categorical_features = test_config["categorical_features"]

        ohe = OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore")
        encoder = ColumnTransformer(transformers=[("ohe", ohe, [i for i, f in enumerate(categorical_features) if f])], remainder="passthrough")
        X_test = encoder.fit_transform(X_test)

        encoder.categories_ = np.array([])
        encoder.categorical_features = categorical_features

        test_embedding = NoEmbedding(test_config, 2, encoder)


if __name__ == "__main__":
    unittest.main()
