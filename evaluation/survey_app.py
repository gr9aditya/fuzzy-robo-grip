"""Anonymous survey-style human evaluation of held-out tactile slip samples."""

from __future__ import annotations

import csv
import uuid
from pathlib import Path

import numpy as np
import streamlit as st

from train_decision_tree import (
    ROOT,
    MamdaniSlipDetector,
    download_dataset,
    grouped_train_validation_test_split,
    load_experiments,
)


DEFAULT_SAMPLE_COUNT = 20
SAMPLE_SEED = 2026
RESPONSE_DIR = ROOT / "results" / "survey_responses"
PREDICTION_PATH = ROOT / "results" / "test_predictions.csv"


def build_survey_cases(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    prediction_rows: list[dict[str, str]],
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> list[dict[str, object]]:
    """Align predictions to held-out samples and select a balanced blind set."""
    if sample_count < 2 or sample_count % 2:
        raise ValueError("The survey sample count must be an even number of at least 2.")
    if len(prediction_rows) != len(test_indices):
        raise ValueError(
            "Prediction rows do not match the held-out test split. "
            "Run train_decision_tree.py again before starting the survey."
        )

    detector = MamdaniSlipDetector().fit(features[train_indices])
    display_features = np.clip(
        (features - detector.feature_min) / (detector.feature_max - detector.feature_min),
        0.0,
        1.0,
    )
    aligned_cases: list[dict[str, object]] = []
    for sample_index, row in zip(test_indices, prediction_rows, strict=True):
        if (
            int(row["true_slip"]) != int(labels[sample_index])
            or row["experiment"] != str(groups[sample_index])
        ):
            raise ValueError(
                "Saved predictions do not match the current dataset split. "
                "Run train_decision_tree.py again before starting the survey."
            )
        aligned_cases.append(
            {
                "sample_index": int(sample_index),
                "experiment": str(groups[sample_index]),
                "features": features[sample_index].tolist(),
                "display_features": display_features[sample_index].tolist(),
                "true_slip": int(labels[sample_index]),
                "decision_tree_prediction": int(row["decision_tree_prediction"]),
                "mamdani_fuzzy_prediction": int(row["mamdani_fuzzy_prediction"]),
            }
        )

    per_class = sample_count // 2
    labels_for_test = np.asarray([case["true_slip"] for case in aligned_cases])
    class_indices = {
        label: np.flatnonzero(labels_for_test == label) for label in (0, 1)
    }
    if any(indices.size < per_class for indices in class_indices.values()):
        raise ValueError(
            f"The held-out test set needs at least {per_class} examples of each class."
        )

    rng = np.random.default_rng(SAMPLE_SEED)
    selected_indices = np.concatenate(
        [
            rng.choice(class_indices[label], size=per_class, replace=False)
            for label in (0, 1)
        ]
    )
    rng.shuffle(selected_indices)
    return [aligned_cases[index] for index in selected_indices]


@st.cache_data(show_spinner="Loading the held-out tactile samples...")
def load_survey_cases(sample_count: int = DEFAULT_SAMPLE_COUNT) -> list[dict[str, object]]:
    """Load the same fixed-seed, experiment-separated test set as the benchmark."""
    if not PREDICTION_PATH.is_file():
        raise FileNotFoundError(
            "Model predictions were not found. Run `python train_decision_tree.py` first."
        )

    dataset_path = download_dataset("spool_solder", ROOT / "data")
    features, labels, groups = load_experiments([dataset_path])
    train_indices, validation_indices, test_indices = grouped_train_validation_test_split(
        labels, groups, random_state=42
    )
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("Training and validation samples unexpectedly overlap.")

    with PREDICTION_PATH.open(newline="", encoding="utf-8") as prediction_file:
        prediction_rows = list(csv.DictReader(prediction_file))
    return build_survey_cases(
        features,
        labels,
        groups,
        train_indices,
        test_indices,
        prediction_rows,
        sample_count,
    )


def save_responses(
    participant_id: str,
    cases: list[dict[str, object]],
    answers: list[dict[str, object]],
) -> Path:
    """Write one anonymous CSV per completed survey to avoid shared-file races."""
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    response_path = RESPONSE_DIR / f"{participant_id}.csv"
    fieldnames = (
        "participant_id",
        "case_number",
        "sample_index",
        "shear_normal_index",
        "shear_change_index",
        "human_answer",
        "confidence_1_to_5",
        "true_slip",
        "decision_tree_prediction",
        "mamdani_fuzzy_prediction",
        "experiment",
    )
    with response_path.open("x", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for case_number, answer in enumerate(answers, start=1):
            case = cases[int(answer["case_index"])]
            writer.writerow(
                {
                    "participant_id": participant_id,
                    "case_number": case_number,
                    "sample_index": case["sample_index"],
                    "shear_normal_index": case["display_features"][0],
                    "shear_change_index": case["display_features"][1],
                    "human_answer": answer["human_answer"],
                    "confidence_1_to_5": answer["confidence"],
                    "true_slip": case["true_slip"],
                    "decision_tree_prediction": case["decision_tree_prediction"],
                    "mamdani_fuzzy_prediction": case["mamdani_fuzzy_prediction"],
                    "experiment": case["experiment"],
                }
            )
    return response_path


def score_answers(
    cases: list[dict[str, object]], answers: list[dict[str, object]]
) -> dict[str, float]:
    """Compute accuracy and balanced accuracy for participants and both models."""
    actual = np.asarray([cases[int(answer["case_index"])]["true_slip"] for answer in answers])
    scores: dict[str, float] = {}
    predictions: dict[str, list[int]] = {
        "participant": [],
        "decision_tree": [],
        "mamdani_fuzzy": [],
    }
    included: list[int] = []
    for answer_index, answer in enumerate(answers):
        case = cases[int(answer["case_index"])]
        predictions["decision_tree"].append(int(case["decision_tree_prediction"]))
        predictions["mamdani_fuzzy"].append(int(case["mamdani_fuzzy_prediction"]))
        if answer["human_answer"] is not None:
            predictions["participant"].append(int(answer["human_answer"]))
            included.append(answer_index)

    for model_name, model_predictions in predictions.items():
        if model_name == "participant":
            model_actual = actual[included]
            model_predicted = np.asarray(model_predictions, dtype=int)
        else:
            model_actual = actual
            model_predicted = np.asarray(model_predictions, dtype=int)
        if model_actual.size == 0:
            scores[f"{model_name}_accuracy"] = float("nan")
            scores[f"{model_name}_balanced_accuracy"] = float("nan")
            continue
        scores[f"{model_name}_accuracy"] = float(
            np.mean(model_actual == model_predicted)
        )
        recalls = [
            float(np.mean(model_predicted[model_actual == label] == label))
            for label in (0, 1)
            if np.any(model_actual == label)
        ]
        scores[f"{model_name}_balanced_accuracy"] = float(np.mean(recalls))
    scores["participant_unsure_rate"] = float(
        sum(answer["human_answer"] is None for answer in answers) / len(answers)
    )
    return scores


def main() -> None:
    st.set_page_config(page_title="Tactile slip survey", page_icon="🖐️", layout="centered")
    st.title("Tactile slip detection: human evaluation")
    st.write(
        "Classify short examples of real tactile-sensor readings as **slip** or "
        "**no slip**. These are recorded samples from the linked robotic-gripper "
        "dataset; this exercise is not a live robot or a glass-filling test."
    )
    st.caption(
        "You will see 20 cases: 10 slip and 10 no-slip samples, selected from "
        "held-out experiments. Labels and model predictions stay hidden until you finish."
    )

    try:
        survey_cases = load_survey_cases()
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        st.error(str(error))
        st.info(
            "Run `python train_decision_tree.py` from the project directory, then reload."
        )
        st.stop()

    if "survey_started" not in st.session_state:
        st.session_state.survey_started = False

    if not st.session_state.survey_started:
        st.subheader("Before you begin")
        st.markdown(
            "- Each case shows two normalized tactile indicators. Higher values mean "
            "higher relative readings compared with training data.\n"
            "- **Shear relative to normal force** compares sideways tactile signal "
            "with normal pressure.\n"
            "- **Change from previous sample** indicates how much the tactile signal "
            "changed since the preceding reading.\n"
            "- Your answers are anonymous. No name, email, or contact details are "
            "requested. After completion, one CSV is saved on the computer/server "
            "running this app.\n"
            "- This is an educational research exercise, not advice for operating "
            "a physical gripper."
        )
        st.warning(
            "The sample intentionally contains equal numbers of slip and no-slip "
            "cases. Survey accuracy does not estimate real-world accuracy at the "
            "dataset's natural slip rate."
        )
        consent = st.checkbox(
            "I understand this is an anonymous evaluation and want to participate."
        )
        if st.button("Start survey", type="primary", disabled=not consent):
            participant_id = uuid.uuid4().hex
            order = np.random.default_rng(int(participant_id[:8], 16)).permutation(
                len(survey_cases)
            )
            st.session_state.participant_id = participant_id
            st.session_state.case_order = order.tolist()
            st.session_state.answers = []
            st.session_state.survey_started = True
            st.rerun()
        return

    participant_id = st.session_state.participant_id
    case_order = st.session_state.case_order
    answers = st.session_state.answers
    current_position = len(answers)

    if current_position < len(case_order):
        case_index = int(case_order[current_position])
        case = survey_cases[case_index]
        display_features = case["display_features"]
        st.progress(current_position / len(case_order))
        st.subheader(f"Case {current_position + 1} of {len(case_order)}")
        st.caption(
            "Values are relative indices from 0 (lower) to 1 (higher) within "
            "the training-data range."
        )
        st.progress(
            float(display_features[0]),
            text=f"Shear relative to normal force: {display_features[0]:.2f}",
        )
        st.progress(
            float(display_features[1]),
            text=f"Change from previous tactile sample: {display_features[1]:.2f}",
        )

        with st.form(key=f"case_form_{participant_id}_{current_position}"):
            human_answer = st.radio(
                "Based on these tactile readings, do you think the object is slipping?",
                options=[1, 0, -1],
                format_func=lambda value: {
                    1: "Slip",
                    0: "No slip",
                    -1: "Not sure",
                }[value],
                index=None,
                horizontal=True,
            )
            confidence = st.slider(
                "How confident are you in this answer?",
                min_value=1,
                max_value=5,
                value=3,
                help="1 = guessing; 5 = very confident",
            )
            submitted = st.form_submit_button(
                "Save answer and continue", type="primary"
            )
        if submitted:
            if human_answer is None:
                st.warning("Choose Slip, No slip, or Not sure before continuing.")
            else:
                answers.append(
                    {
                        "case_index": case_index,
                        "human_answer": None if human_answer == -1 else human_answer,
                        "confidence": confidence,
                    }
                )
                st.session_state.answers = answers
                st.rerun()
        return

    st.progress(1.0)
    st.subheader("Survey complete")
    if not st.session_state.get("responses_saved", False):
        try:
            response_path = save_responses(participant_id, survey_cases, answers)
        except OSError as error:
            st.error(f"Could not save your responses: {error}")
        else:
            st.session_state.responses_saved = True
            st.success(f"Anonymous responses saved to `{response_path}`.")

    summary = score_answers(survey_cases, answers)
    unsure_rate = summary["participant_unsure_rate"]
    participant_accuracy = summary["participant_accuracy"]
    st.write(
        f"You selected **Not sure** for {unsure_rate:.0%} of cases. "
        f"Accuracy on your definite answers: "
        f"**{participant_accuracy:.0%}**."
    )
    st.caption(
        "The table below reveals the reference labels and model answers. "
        "A human ‘Not sure’ answer is excluded from human accuracy."
    )
    comparison_rows = []
    for answer_number, answer in enumerate(answers, start=1):
        case = survey_cases[int(answer["case_index"])]
        comparison_rows.append(
            {
                "Case": answer_number,
                "Your answer": (
                    "Not sure"
                    if answer["human_answer"] is None
                    else "Slip"
                    if answer["human_answer"] == 1
                    else "No slip"
                ),
                "Reference label": "Slip" if case["true_slip"] else "No slip",
                "Decision tree": (
                    "Slip" if case["decision_tree_prediction"] else "No slip"
                ),
                "Mamdani fuzzy": (
                    "Slip" if case["mamdani_fuzzy_prediction"] else "No slip"
                ),
                "Confidence": answer["confidence"],
            }
        )
    st.dataframe(comparison_rows, width="stretch", hide_index=True)

    comparison = st.columns(3)
    for column, model_name, label in zip(
        comparison,
        ("participant", "decision_tree", "mamdani_fuzzy"),
        ("Your accuracy", "Decision-tree accuracy", "Mamdani accuracy"),
        strict=True,
    ):
        column.metric(label, f"{summary[f'{model_name}_accuracy']:.0%}")
    if st.button("Start a new anonymous session"):
        for key in (
            "survey_started",
            "participant_id",
            "case_order",
            "answers",
            "responses_saved",
        ):
            st.session_state.pop(key, None)
        st.rerun()


if __name__ == "__main__":
    main()
