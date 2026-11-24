import unittest

import numpy as np

from survey_app import build_survey_cases, score_answers


class SurveyAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.arange(80, dtype=float).reshape(40, 2)
        self.labels = np.tile([0, 1], 20)
        self.groups = np.repeat([f"experiment-{index}" for index in range(10)], 4)
        self.train_indices = np.arange(0, 20)
        self.test_indices = np.arange(20, 40)
        self.prediction_rows = [
            {
                "experiment": str(self.groups[index]),
                "true_slip": str(self.labels[index]),
                "decision_tree_prediction": str(self.labels[index]),
                "mamdani_fuzzy_prediction": str(1 - self.labels[index]),
            }
            for index in self.test_indices
        ]

    def test_build_survey_cases_selects_balanced_examples(self) -> None:
        cases = build_survey_cases(
            self.features,
            self.labels,
            self.groups,
            self.train_indices,
            self.test_indices,
            self.prediction_rows,
            sample_count=10,
        )

        self.assertEqual(len(cases), 10)
        self.assertEqual(sum(int(case["true_slip"]) for case in cases), 5)
        self.assertEqual(
            len({int(case["sample_index"]) for case in cases}),
            len(cases),
        )
        self.assertTrue(
            all(
                int(case["sample_index"]) in self.test_indices
                for case in cases
            )
        )

    def test_build_survey_cases_rejects_stale_predictions(self) -> None:
        stale_rows = [dict(row) for row in self.prediction_rows]
        stale_rows[0]["true_slip"] = "9"
        with self.assertRaisesRegex(ValueError, "do not match"):
            build_survey_cases(
                self.features,
                self.labels,
                self.groups,
                self.train_indices,
                self.test_indices,
                stale_rows,
                sample_count=10,
            )

    def test_score_answers_excludes_not_sure_from_human_accuracy(self) -> None:
        cases = [
            {"true_slip": 0, "decision_tree_prediction": 0, "mamdani_fuzzy_prediction": 1},
            {"true_slip": 1, "decision_tree_prediction": 1, "mamdani_fuzzy_prediction": 0},
            {"true_slip": 1, "decision_tree_prediction": 0, "mamdani_fuzzy_prediction": 1},
        ]
        answers = [
            {"case_index": 0, "human_answer": 0, "confidence": 4},
            {"case_index": 1, "human_answer": None, "confidence": 2},
            {"case_index": 2, "human_answer": 1, "confidence": 3},
        ]

        scores = score_answers(cases, answers)

        self.assertEqual(scores["participant_accuracy"], 1.0)
        self.assertAlmostEqual(scores["participant_unsure_rate"], 1 / 3)
        self.assertAlmostEqual(scores["decision_tree_accuracy"], 2 / 3)
        self.assertAlmostEqual(scores["mamdani_fuzzy_accuracy"], 1 / 3)

    def test_sample_count_must_be_even(self) -> None:
        with self.assertRaisesRegex(ValueError, "even"):
            build_survey_cases(
                self.features,
                self.labels,
                self.groups,
                self.train_indices,
                self.test_indices,
                self.prediction_rows,
                sample_count=5,
            )


if __name__ == "__main__":
    unittest.main()
