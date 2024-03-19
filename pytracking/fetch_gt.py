import os
import sys
import argparse
import numpy as np
import yaml

env_path = os.path.join(os.path.dirname(__file__), "..")
if env_path not in sys.path:
    sys.path.append(env_path)

GT_FILE_NAME = "labels_events_left.npy"
DTYPE = dtype = np.dtype(
    [
        ("frame", "<u8"),
        ("track_id", "<u4"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("w", "<f4"),
        ("h", "<f4"),
        ("class_confidence", "<f4"),
        ("class_id", "u1"),
        ("visibility", "<f4"),
    ]
)


def fetch_gt_for_sequence(gt: np.ndarray, frame_offset: int, debug: int = 0) -> np.ndarray:
    """Fetch the ground truth for a sequence.
    args:
        gt: Ground truth for all sequences.
        frame_offset: Offset for the sequence.
    returns:
        gt_sequence: Ground truth for the sequence.
    """
    track_ids = np.unique(gt["track_id"])
    filtered_labels = np.zeros((len(gt),), dtype=DTYPE)
    idx_offset = 0
    for track_id in track_ids:
        if debug > 0:
            diff = np.diff(gt[gt["track_id"] == track_id]["frame"])
            if any(diff > 1):
                print(f"Track {track_id} has missing frames at {np.where(diff > 1)}.")
        initial_frame = gt[gt["track_id"] == track_id][0]["frame"]
        track_labels = gt[gt["track_id"] == track_id]
        filtered_track_labels = track_labels[track_labels["frame"] >= frame_offset + initial_frame]
        filtered_labels[idx_offset : idx_offset + len(filtered_track_labels)] = filtered_track_labels
        idx_offset = idx_offset + len(filtered_track_labels)
    pre_filter = len(filtered_labels)
    filtered_labels = filtered_labels[filtered_labels[:] != np.array((0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=DTYPE)]
    post_filter = len(filtered_labels)
    if debug > 0:
        if (pre_filter - post_filter) % frame_offset != 0:
            print(f"Uncommon umber of filteredl labels: {pre_filter - post_filter} labels.")
    sorted_index = np.argsort(filtered_labels, order=["frame", "track_id"])
    filtered_labels_sorted = filtered_labels[sorted_index]

    filtered_labels_sorted["track_id"] = filtered_labels_sorted["track_id"] + 1

    return filtered_labels_sorted


def main():
    parser = argparse.ArgumentParser(description="Fetch ground truth files and filter out frames if necessary.")
    parser.add_argument("ds_root", type=str, help="Dataset root.")
    parser.add_argument("frame_offset", type=int, help="Offset for the sequence.")
    parser.add_argument("sequences", type=str, help="Path to yaml file with sequences.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--debug", type=int, default=0, help="Debug level.")
    args = parser.parse_args()

    sequences = yaml.safe_load(open(args.sequences, "r"))["sequences"]

    for sequence in sequences:
        print(f"Processing sequence {sequence}.")
        gt_path = os.path.join(args.ds_root, sequence, GT_FILE_NAME)

        if os.path.exists(gt_path):
            gt = np.load(gt_path)
            filtered_gt = fetch_gt_for_sequence(gt, args.frame_offset, args.debug)
            output_path = os.path.join(args.output_dir, sequence, f"labels.npy")
            if os.path.exists(os.path.dirname(output_path)):
                np.save(output_path, filtered_gt)
            else:
                print(f"Output directory {os.path.dirname(output_path)} does not exist.")
        else:
            print(f"Ground truth file {gt_path} does not exist.")
            continue


if __name__ == "__main__":
    main()
