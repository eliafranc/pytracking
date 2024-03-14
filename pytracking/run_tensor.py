import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), "..")
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker

"""
log:
    - 2024_01_10_120347_10km_001
    - 2024_01_10_153109_dji_000
    - 2024_01_10_112814_drone_000
    - 2024_01_10_162004_focus_000 (2 drones, 1 barely visible, good results)
"""

PATH_TO_DATA = "/home/efranc/data"
TIMINGS = "frames_ts.csv"
RGB_FRAME_DIR = "frames"
EVENT_FILE = "events_left_final.h5"
HOMOGRAPHY_FILE = "Projection_rgb_to_events_left.npy"
LABEL_FILE = "labels_events_left.npy"


def run_tensor(
    tracker_name,
    tracker_param,
    timings_file,
    rgb_frame_dir,
    event_file,
    homography_file,
    label_file,
    debug=None,
    vis=False,
    rgb_only=False,
    save_results=False,
):
    """Run the tracker on an input tensor that is mixed rgb and event information.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        timings_file: Path to a csv file with timestamps.
        rgb_frame_dir: Path to a directory with rgb frames.
        event_file: Path to a hdf5 event file.
        homography_file: Path to a numpy homography file.
        label_file: Path to a numpy label file.
        vis: Visualize each frame with the predicted bounding boxes.
        rgb_only: Use only RGB frames without any event information.
        save_results: Save the results in a numpy file.
    """
    tracker = Tracker(tracker_name, tracker_param)
    tracker.run_on_tensor(
        timings_file,
        rgb_frame_dir,
        event_file,
        homography_file,
        label_file,
        debug=debug,
        vis=vis,
        rgb_only=rgb_only,
        save_results=save_results,
    )


def main():

    parser = argparse.ArgumentParser(description="Run the tracker on a custom input tensor.")
    parser.add_argument("tracker_name", type=str, help="Name of tracking method.")
    parser.add_argument("tracker_param", type=str, help="Name of parameter file.")
    parser.add_argument("sequence", type=str, help="Path to a sequence firectory.")
    parser.add_argument("--debug", type=int, default=0, help="Debug level.")
    parser.add_argument("--visualize", dest="visualize", action="store_true", help="Visualize bounding boxes")
    parser.set_defaults(visualize=False)
    parser.add_argument("--rgb_only", dest="rgb_only", action="store_true", help="Use only RGB frames")
    parser.set_defaults(rgb_only=False)
    parser.add_argument("--save_results", dest="save_results", action="store_true", help="Visualize bounding boxes")
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_tensor(
        args.tracker_name,
        args.tracker_param,
        f"{PATH_TO_DATA}/{args.sequence}/{TIMINGS}",
        f"{PATH_TO_DATA}/{args.sequence}/{RGB_FRAME_DIR}",
        f"{PATH_TO_DATA}/{args.sequence}/{EVENT_FILE}",
        f"{PATH_TO_DATA}/{args.sequence}/{HOMOGRAPHY_FILE}",
        f"{PATH_TO_DATA}/{args.sequence}/{LABEL_FILE}",
        debug=args.debug,
        vis=args.visualize,
        rgb_only=args.rgb_only,
        save_results=args.save_results,
    )


if __name__ == "__main__":
    main()
