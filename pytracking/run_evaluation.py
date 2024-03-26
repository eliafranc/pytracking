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


# track_id_2_name = {1: 'rgb', 2: '5ms', 3: '10ms'}
track_id_2_name = {1: "RGB", 2: "RGB + EV 5ms", 3: "RGB + EV 10ms", 4: "RGB + EV 2ms"}


def run_evaluation(
    tracker_name,
    tracker_param,
    report_name,
    plot_type=("success"),
    dataset_name="evdrone",
    sequence_name=None,
    print_result=False,
    empty_pred_zero=True,
):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        report_name: Name of the report that is being run.
        plot_type: List of scores to display. Can contain 'success',
                'prec' (precision), and 'norm_prec' (normalized precision).
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence_name: Sequence name. Can be set if only one sequence should be evaluated.
        print_results: Print results.

    """

    trackers = [
        Tracker(tracker_name, tracker_param, run_id, disp_name) for run_id, disp_name in track_id_2_name.items()
    ]
    dataset = get_dataset(dataset_name)

    if sequence_name is not None:
        dataset = [dataset[sequence_name]]

    plot_results(trackers, dataset, report_name, plot_types=plot_type, skip_missing_seq=empty_pred_zero)

    if print_result:
        print_results(trackers, dataset, report_name, plot_types=plot_type, skip_missing_seq=empty_pred_zero)


def main():
    parser = argparse.ArgumentParser(description="Run tracker on sequence or dataset.")
    parser.add_argument("tracker_name", type=str, help="Name of tracking method.")
    parser.add_argument("tracker_param", type=str, help="Name of parameter file.")
    parser.add_argument("report_name", type=str, help="Name for the output directory.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evdrone",
        help="Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).",
    )
    parser.add_argument(
        "--plot_type",
        nargs="*",
        default=("success"),
        help="List of scores to display. Can contain 'success', 'prec' (precision), and 'norm_prec' (normalized precision).",
    )
    parser.add_argument("--sequence", type=str, default=None, help="Sequence name.")
    parser.add_argument("--print_results", action="store_true", help="Print results.")
    parser.add_argument("--empty_pred_zero", action="store_true", help="Set empty prediction to zero.")

    args = parser.parse_args()

    run_evaluation(
        args.tracker_name,
        args.tracker_param,
        args.report_name,
        args.plot_type,
        args.dataset,
        args.sequence,
        args.print_results,
        args.empty_pred_zero,
    )


if __name__ == "__main__":
    main()
