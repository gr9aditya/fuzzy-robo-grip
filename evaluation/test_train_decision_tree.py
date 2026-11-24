import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from train_decision_tree import (
    MamdaniSlipDetector,
    features_from_experiment,
    grouped_train_validation_test_split,
    load_experiments,
)


class TactileSlipPipelineTests(unittest.TestCase):
    def test_feature_extraction_returns_finite_interpretable_features(self) -> None:
        raw_data = np.ones((18, 3, 4), dtype=float)
        raw_data[:, 0, :] = np.array([1.0, 2.0, 3.0, 4.0])

        features = features_from_experiment(raw_data)

        self.assertEqual(features.shape, (4, 2))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features[:, 0] > 0))
        self.assertEqual(features[0, 1], 0.0)

    def test_hdf5_loader_maps_slip_labels_and_drops_other_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "fixture.h5"
            with h5py.File(path, "w") as handle:
                experiment = handle.create_group("object/Pose1/Exp1")
                raw_data = np.ones((18, 3, 6), dtype=float)
                raw_data[:, 0, :] = np.arange(1, 7)
                experiment.create_dataset("tactile_data_raw", data=raw_data)
                experiment.create_dataset(
                    "tactile_slips_label", data=np.array([0, 1, 4, 5, 2, 6])
                )

            features, labels, groups = load_experiments([path])

        np.testing.assert_array_equal(labels, [0, 1, 0, 1])
        self.assertEqual(features.shape, (4, 2))
        self.assertEqual(np.unique(groups).size, 1)

    def test_mamdani_scores_increase_for_high_shear_features(self) -> None:
        training_features = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0]])
        detector = MamdaniSlipDetector().fit(training_features)

        scores = detector.predict_score(
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        )

        self.assertEqual(scores.shape, (3,))
        self.assertLess(scores[0], 0.5)
        self.assertAlmostEqual(scores[1], scores[0], places=2)
        self.assertGreater(scores[2], scores[0])

    def test_grouped_split_keeps_experiments_disjoint(self) -> None:
        labels = np.tile([0, 1], 20)
        groups = np.repeat([f"experiment-{index}" for index in range(20)], 2)

        train, validation, test = grouped_train_validation_test_split(
            labels, groups, random_state=42
        )

        train_groups = set(groups[train])
        validation_groups = set(groups[validation])
        test_groups = set(groups[test])
        self.assertFalse(train_groups & validation_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(validation_groups & test_groups)


if __name__ == "__main__":
    unittest.main()
