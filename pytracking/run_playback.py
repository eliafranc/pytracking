import argparse
import importlib
import os
import sys

env_path = os.path.join(os.path.dirname(__file__), "..")
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.analysis.playback_results import playback_results
from pytracking.evaluation import Tracker
from pytracking.evaluation.datasets import get_dataset

track_id_2_name = {
    1: "RGB",
    2: "RGB + EV 1ms",
    3: "RGB + EV 2ms",
    4: "RGB + EV 5ms",
    5: "RGB + EV 10ms",
    6: "RGB + EV 15ms",
    7: "RGB + EV 20ms",
}


def run_playback(trackers, sequence):
    playback_results(trackers, sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tracker on sequence or dataset.")
    parser.add_argument("sequence", help="Sequence name.")
    args = parser.parse_args()

    ds = get_dataset("evdrone")
    trackers = [Tracker("rts", "rts50", run_id, disp_name) for run_id, disp_name in track_id_2_name.items()]
    trackers.extend(Tracker("tomp", "tomp50", run_id, disp_name) for run_id, disp_name in track_id_2_name.items())
    sequence = ds[args.sequence]

    run_playback(trackers, sequence)
