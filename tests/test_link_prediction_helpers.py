import unittest

import torch

from run_louvainne_experiments import (
    build_link_prediction_embeddings,
    prepare_train_link_prediction_edges,
)


class DummyRunner:
    def __init__(self, embeddings: torch.Tensor) -> None:
        self._embeddings = embeddings

    def embed(self, edge_weights, seed: int) -> torch.Tensor:
        return self._embeddings.clone()


class LinkPredictionHelperTests(unittest.TestCase):
    def test_prepare_train_link_prediction_edges_keeps_attribute_only_predictions(self) -> None:
        structure_edges = {
            (0, 1): 1.0,
            (1, 2): 1.0,
        }
        predicted_edges = {
            (0, 1): 0.2,
            (0, 2): 0.9,
            (2, 3): 0.8,
        }
        train_edge_index = torch.tensor(
            [
                [0, 1, 1, 2],
                [1, 0, 2, 1],
            ],
            dtype=torch.long,
        )
        val_pos = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
        test_pos = torch.tensor([[1, 3], [3, 1]], dtype=torch.long)

        train_structure, train_predicted, train_fused = prepare_train_link_prediction_edges(
            train_edge_index=train_edge_index,
            val_pos=val_pos,
            test_pos=test_pos,
            structure_edges=structure_edges,
            predicted_edges=predicted_edges,
            overlap_scale=1.0,
            new_scale=0.75,
        )

        self.assertEqual(train_structure, structure_edges)
        self.assertNotIn((0, 2), train_predicted)
        self.assertIn((2, 3), train_predicted)
        self.assertEqual(train_predicted[(2, 3)], 0.8)
        self.assertEqual(train_fused[(0, 1)], 1.2)
        self.assertEqual(train_fused[(2, 3)], 0.6)

    def test_build_link_prediction_embeddings_respects_attention_and_feature_flags(self) -> None:
        base_embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        runner = DummyRunner(base_embeddings)
        fused_edges = {
            (0, 1): 1.0,
            (1, 2): 1.0,
        }
        predicted_edges = {
            (0, 2): 0.9,
        }
        feature_matrix = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        baseline_embeddings = build_link_prediction_embeddings(
            runner=runner,
            fused_edges=fused_edges,
            predicted_edges={},
            seed=7,
            feature_matrix=None,
            feature_dim=0,
            attention_gamma=0.0,
            attention_temperature=1.0,
        )
        improved_embeddings = build_link_prediction_embeddings(
            runner=runner,
            fused_edges=fused_edges,
            predicted_edges=predicted_edges,
            seed=7,
            feature_matrix=feature_matrix,
            feature_dim=2,
            attention_gamma=0.5,
            attention_temperature=1.0,
        )

        self.assertEqual(baseline_embeddings.shape, (3, 2))
        self.assertEqual(improved_embeddings.shape, (3, 4))
        self.assertTrue(torch.allclose(baseline_embeddings, torch.nn.functional.normalize(base_embeddings, dim=1)))
        self.assertFalse(torch.allclose(improved_embeddings[:, :2], baseline_embeddings))


if __name__ == "__main__":
    unittest.main()
