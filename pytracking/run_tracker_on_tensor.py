import os
import sys
import argparse
import yaml

env_path = os.path.join(os.path.dirname(__file__), "..")
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.running import run_tensor_dataset
from pytracking.evaluation import Tracker


# run_arguments = {
#     1: {"rgb_only": True, "delta_t": 10},
#     3: {"rgb_only": False, "delta_t": 2},

# }
run_arguments = {
    1: {"rgb_only": True, "delta_t": 10},
    2: {"rgb_only": False, "delta_t": 1},
    3: {"rgb_only": False, "delta_t": 2},
    4: {"rgb_only": False, "delta_t": 5},
    5: {"rgb_only": False, "delta_t": 10},
    6: {"rgb_only": False, "delta_t": 15},
    7: {"rgb_only": False, "delta_t": 20},
}


def run_tracker_on_tensor(
    tracker_name,
    tracker_param,
    delta_t=10,
    rgb_only=False,
    run_id=None,
    sequence=None,
    debug=0,
    threads=0,
    visdom_info=None,
):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """

    visdom_info = {} if visdom_info is None else visdom_info

    dataset = yaml.safe_load(open("experiments/ststephan_sequences.yaml"))["sequences"]

    if sequence is not None:
        dataset = [sequence]

    trackers = [Tracker(tracker_name, tracker_param, run_id)]

    run_tensor_dataset(dataset, trackers, delta_t, rgb_only, debug, threads, visdom_info=visdom_info)


def main():
    parser = argparse.ArgumentParser(description="Run tracker on sequence or dataset.")
    parser.add_argument("tracker_name", type=str, help="Name of tracking method.")
    parser.add_argument("tracker_param", type=str, help="Name of parameter file.")
    parser.add_argument("--delta_t", type=int, default=10, help="Time delta for events.")
    parser.add_argument("--rgb_only", action="store_true", help="Use only RGB frames")
    parser.add_argument("--runid", type=int, default=None, help="The run id.")
    parser.add_argument("--sequence", type=str, default=None, help="Sequence number or name.")
    parser.add_argument("--debug", type=int, default=0, help="Debug level.")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads.")
    parser.add_argument("--use_visdom", type=bool, default=True, help="Flag to enable visdom.")
    parser.add_argument("--visdom_server", type=str, default="127.0.0.1", help="Server for visdom.")
    parser.add_argument("--visdom_port", type=int, default=8097, help="Port for visdom.")

    args = parser.parse_args()

    if args.sequence is not None:
        try:
            seq_name = int(args.sequence)
        except:
            seq_name = args.sequence
    else:
        seq_name = None

    for run_id, run_args in run_arguments.items():
        run_tracker_on_tensor(
            args.tracker_name,
            args.tracker_param,
            run_args["delta_t"],
            run_args["rgb_only"],
            run_id,
            seq_name,
            args.debug,
            args.threads,
            {"use_visdom": args.use_visdom, "server": args.visdom_server, "port": args.visdom_port},
        )


if __name__ == "__main__":
    main()
