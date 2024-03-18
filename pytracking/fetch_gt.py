import os
import sys
import argparse
import numpy as np

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


def fetch_gt_for_sequence(gt: np.ndarray, frame_offset: int):
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
        # print(gt[gt['track_id'] == track_id])
        diff = np.diff(gt[gt["track_id"] == track_id]["frame"])
        if any(diff > 1):
            print(f"Track {track_id} has missing frames at {np.where(diff > 1)}.")
        initial_frame = gt[gt["track_id"] == track_id][0]["frame"]
        track_labels = gt[gt["track_id"] == track_id]
        filtered_track_labels = track_labels[track_labels["frame"] >= frame_offset + initial_frame]
        filtered_labels[idx_offset : idx_offset + len(filtered_track_labels)] = filtered_track_labels
        idx_offset = len(filtered_track_labels)
    pre_filter = len(filtered_labels)
    filtered_labels = filtered_labels[filtered_labels[:] != np.array((0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=DTYPE)]
    post_filter = len(filtered_labels)
    if (pre_filter - post_filter) % 4 != 0:
        print(f"Filtered out {pre_filter - post_filter} labels.")
    sorted_index = np.argsort(filtered_labels, order=["frame", "track_id"])
    filtered_labels_sorted = filtered_labels[sorted_index]

    return filtered_labels_sorted


def main():
    parser = argparse.ArgumentParser(description="Fetch ground truth files and filter out frames if necessary.")
    parser.add_argument("ds_root", type=str, help="Dataset root.")
    parser.add_argument("frame_offset", type=int, help="Offset for the sequence.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    args = parser.parse_args()

    for sequence in os.listdir(args.ds_root):
        gt_path = os.path.join(args.ds_root, sequence, GT_FILE_NAME)
        print(sequence)

        if os.path.exists(gt_path):
            gt = np.load(gt_path)
            filtered_gt = fetch_gt_for_sequence(gt, args.frame_offset)
            output_path = os.path.join(args.output_dir, sequence, f"labels.npy")
            if os.path.exists(os.path.dirname(output_path)):
                np.save(output_path, filtered_gt)
            # else:
            #     print(f"Output directory {os.path.dirname(output_path)} does not exist.")
        else:
            print(f"Ground truth file {gt_path} does not exist.")
            continue


if __name__ == "__main__":
    main()
