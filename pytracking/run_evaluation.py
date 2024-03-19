import os
import sys
import argparse
import numpy as np
import yaml

env_path = os.path.join(os.path.dirname(__file__), "..")
if env_path not in sys.path:
    sys.path.append(env_path)

from prophesee.metrics.coco_eval import evaluate_detection
from prophesee.io.box_loading import to_prophesee


prediction_type_suffix = {"rgb": "rgb", "5": "5ms", "10": "10ms"}


def run_evaluation_on_sequence(gt_path, pred_path, save_evaluation=False):
    """Run the evaluation of the tracker output on a single sequence.
    args:
        gt_path: Path to the ground truth file.
        pred_path: Path to the prediction file.
    """
    gt_list = []
    pred_list = []
    gt = np.load(gt_path)
    pred = np.load(pred_path)
    gt, pred = to_prophesee(gt, pred)
    timestamps = np.unique(gt["t"])
    for ts in timestamps:
        gt_list.append(gt[gt["t"] == ts])
        pred_list.append(pred[pred["t"] == ts])

    return evaluate_detection(gt_list, pred_list, return_aps=save_evaluation)


def run_evaluation(results_root, pred_type, save_evaluation=False):
    """Run the evaluation of the tracker output.
    args:
        results_root: Root directory of the results.
        output_file: Path to output file.
    """
    assert (
        pred_type in prediction_type_suffix.keys()
    ), f"Invalid prediction type. Must be one of {prediction_type_suffix.keys()}."

    for sequence in os.listdir(results_root):
        output = os.path.join(results_root, sequence, f"evaluation_{prediction_type_suffix.get(pred_type)}.yml")

        gt_path = os.path.join(results_root, sequence, "labels.npy")
        pred_path = os.path.join(results_root, sequence, f"predictions_{prediction_type_suffix.get(pred_type)}.npy")
        metrics = run_evaluation_on_sequence(gt_path, pred_path, save_evaluation)

        if metrics is not None:
            print(metrics)
            with open(output, "w") as f:
                yaml.dump(metrics, f)


def main():
    parser = argparse.ArgumentParser(description="Run the evaluation.")
    parser.add_argument("results_root", type=str, help="Root directory of the results.")
    parser.add_argument(
        "pred_type",
        type=str,
        help="Which prediction should be used for evaluation. Must be 'rgb', or number of ms of combined event and rgb frame, i.e. '5', '10'.",
    )
    parser.add_argument("--save_evaluation", action="store_true", help="Save the evaluation results.")
    parser.set_defaults(save_evaluation=False)
    args = parser.parse_args()

    run_evaluation(args.results_root, args.pred_type, args.save_evaluation)


if __name__ == "__main__":
    main()
