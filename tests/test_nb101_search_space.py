import unittest
import torch
import os
import numpy as np
from naslib.search_spaces import NasBench101SearchSpace
from naslib.search_spaces.nasbench101 import conversions
import naslib.utils.nb101_api as api

SPEC = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 3, 3, 3, 1)
SAMPLED_SPEC = (0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 2, 3, 3, 1)
HASHSPEC = 'ff940d27cbe9ed0a639a68a9c3f87283'  # Random HASH from NB101
FIXED_ARCH = (0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 3, 3, 4, 1)


def create_dummy_api():
    # TODO: Complete
    nb101_datapath = os.path.join(os.getcwd(), "assets", "nb101_dummy.pkl")
    nb101_data = api.NASBench(nb101_datapath)

    return {"api": api, "nb101_data": nb101_data}


def create_model(spec, n_classes=10):
    torch.manual_seed(9001)
    graph = NasBench101SearchSpace(n_classes=n_classes)
    graph.set_spec(spec)
    return graph


class NasBench101SearchSpaceTest(unittest.TestCase):

    def test_set_and_get_spec(self):
        graph = NasBench101SearchSpace()
        graph.set_spec(SPEC)
        retrieved_spec = graph.get_hash()
        self.assertEqual(SPEC, retrieved_spec)

    def test_set_and_get_spec_hash(self):
        graph = NasBench101SearchSpace()
        dummy_api = create_dummy_api()
        graph.set_spec(HASHSPEC, dummy_api)
        retrieved_spec = graph.get_hash()
        print(retrieved_spec)
        self.assertEqual(FIXED_ARCH, retrieved_spec)

    def test_forward_pass(self):
        torch.manual_seed(9001)
        graph = create_model(n_classes=10, spec=SPEC)

        out = graph(torch.randn(3, 3, 32, 32))
        self.assertTrue(torch.allclose(out[0].detach(), torch.tensor([0.0737, 0.0128, 0.0086, 0.0214, 0.0912, -0.0532,
                                                                      0.0479, 0.1870,
                                                                      -0.0248, 0.1075]), rtol=1e-2))
        self.assertTrue(torch.allclose(out[1].detach(), torch.tensor([0.0667, 0.0115, 0.0107, 0.0106, 0.0570, -0.0186,
                                                                      0.0353, 0.1650,
                                                                      -0.0165, 0.0848]), rtol=1e-2))
        self.assertTrue(torch.allclose(out[2].detach(), torch.tensor([0.0536, 0.0136, -0.0104, 0.0174, 0.0753, -0.0871,
                                                                      0.0472, 0.1974,
                                                                      -0.0231, 0.1336]), rtol=1e-2))
        self.assertEqual(out.shape, (3, 10))

    # TODO: Complete. These tests require a dummy NAS-Bench-101 API.
    def test_sample_random_architecture(self):
        graph = NasBench101SearchSpace()
        np.random.seed(9001)
        graph.sample_random_architecture(create_dummy_api())
        spec = graph.get_hash()

        self.assertEqual(spec, SAMPLED_SPEC)

    def test_query(self):
        graph = NasBench101SearchSpace()
        graph.set_spec(SPEC)

    def test_get_arch_iterator(self):
        graph = NasBench101SearchSpace()
        it = graph.get_arch_iterator(create_dummy_api())

        archs = set(it)

        self.assertEqual(len(archs), 30)
        self.assertIn('ff97db031fa41552d437b079b2befd80', archs)
        self.assertIn('ff968f800464555f97c776c71481826d', archs)
        self.assertIn('ff940d27cbe9ed0a639a68a9c3f87283', archs)

    def test_mutate(self):
        graph_parent = create_model(spec=SPEC)
        graph_child = NasBench101SearchSpace()

        graph_child.mutate(graph_parent, create_dummy_api())

        parent_spec = graph_parent.get_hash()
        child_spec = graph_child.get_hash()

        self.assertNotEqual(parent_spec, child_spec)

        out = graph_child(torch.randn(3, 3, 32, 32))
        self.assertEqual(out.shape, (3, 10))

    def test_get_nbhd(self):
        graph = create_model(spec=SPEC)
        neighbours = graph.get_nbhd(create_dummy_api())
        print(len(neighbours))

        self.assertEqual(len(neighbours), 12)

    def test_conversions(self):
        torch.manual_seed(9001)
        data = torch.randn(3, 3, 32, 32)

        graph1 = create_model(spec=SPEC)
        out1 = graph1(data)

        spec = conversions.convert_tuple_to_spec(SPEC)
        graph2 = create_model(spec=spec)
        out2 = graph2(data)

        tup = conversions.convert_spec_to_tuple(spec)
        graph3 = create_model(spec=tup)
        out3 = graph3(data)

        self.assertEqual(SPEC, tup)
        self.assertTrue(torch.allclose(out1, out2))
        self.assertTrue(torch.allclose(out1, out3))
        self.assertTrue(torch.allclose(out2, out3))


if __name__ == '__main__':
    unittest.main()
