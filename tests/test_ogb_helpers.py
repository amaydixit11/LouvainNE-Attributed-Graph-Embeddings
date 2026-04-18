import unittest

import torch

from benchmark_ogb import build_structure_reweighted_edges


class OGBHelperTests(unittest.TestCase):
    def test_build_structure_reweighted_edges_only_reweights_observed_edges(self) -> None:
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2],
            ],
            dtype=torch.long,
        )
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.0, 0.8],
            ],
            dtype=torch.float32,
        )

        fused_edges, predicted_edges = build_structure_reweighted_edges(
            edge_index=edge_index,
            normalized_features=features,
            min_similarity=0.75,
            overlap_scale=1.0,
            batch_size=2,
        )

        self.assertEqual(set(fused_edges.keys()), {(0, 1), (1, 2), (2, 3)})
        self.assertIn((0, 1), predicted_edges)
        self.assertIn((2, 3), predicted_edges)
        self.assertNotIn((1, 2), predicted_edges)
        self.assertGreater(fused_edges[(0, 1)], 1.0)
        self.assertEqual(fused_edges[(1, 2)], 1.0)


if __name__ == "__main__":
    unittest.main()
