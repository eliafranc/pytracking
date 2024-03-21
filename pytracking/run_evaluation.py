import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), "..")
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.environment import env_settings
from pytracking.analysis.plot_results import plot_results, print_results
from pytracking.evaluation.datasets import get_dataset
from pytracking.evaluation import Sequence, Tracker


def run_evaluation(
    tracker_name, tracker_param, input_type, dataset_name="evdrone", sequence_name=None, print_result=False
):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        input_type: Type of input (rgb, 10ms, 5ms etc.).
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence_name: Sequence name. Can be set if only one sequence should be evaluated.
        print_results: Print results.

    """

    trackers = [Tracker(tracker_name, tracker_param)]
    dataset = get_dataset(dataset_name)

    if sequence_name is not None:
        dataset = [dataset[sequence_name]]

    plot_results(trackers, dataset, input_type)

    if print_result:
        print_results(trackers, dataset, input_type)


def main():
    parser = argparse.ArgumentParser(description="Run tracker on sequence or dataset.")
    parser.add_argument("tracker_name", type=str, help="Name of tracking method.")
    parser.add_argument("tracker_param", type=str, help="Name of parameter file.")
    parser.add_argument("input_type", type=str, help="Type of input (rgb, 10ms, 5ms etc.).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evdrone",
        help="Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).",
    )
    parser.add_argument("--sequence", type=str, default=None, help="Sequence name.")
    parser.add_argument("--print_results", action="store_true", help="Print results.")

    args = parser.parse_args()

    run_evaluation(
        args.tracker_name, args.tracker_param, args.input_type, args.dataset, args.sequence, args.print_results
    )


if __name__ == "__main__":
    main()
