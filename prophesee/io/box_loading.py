# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Defines some tools to handle events.
In particular :
    -> defines events' types
    -> defines functions to read events from binary .dat files using numpy
    -> defines functions to write events to binary .dat files using numpy
"""

from __future__ import print_function
import numpy as np

BBOX_DTYPE = np.dtype(
    {
        "names": ["t", "x", "y", "w", "h", "class_id", "track_id", "class_confidence"],
        "formats": ["<i8", "<f4", "<f4", "<f4", "<f4", "<u4", "<u4", "<f4"],
        "offsets": [0, 8, 12, 16, 20, 24, 28, 32],
        "itemsize": 40,
    }
)


def calculate_padding(w, h):
    pad_w = int(2 * 0.08 * w)
    pad_h = int(2 * 0.1 * h)
    pad_x = round(pad_w / 2)
    pad_y = round(pad_h * 0.8)
    return {"x": pad_x, "y": pad_y, "w": pad_w, "h": pad_h}


def reformat_boxes(boxes):
    """ReFormat boxes according to new rule
    This allows to be backward-compatible with imerit annotation.
        't' = 'ts'
        'class_confidence' = 'confidence'
    """
    if "t" not in boxes.dtype.names or "class_confidence" not in boxes.dtype.names:
        new = np.zeros((len(boxes),), dtype=BBOX_DTYPE)
        for name in boxes.dtype.names:
            if name == "ts":
                new["t"] = boxes[name]
            elif name == "confidence":
                new["class_confidence"] = boxes[name]
            else:
                new[name] = boxes[name]
        return new
    else:
        return boxes


def to_prophesee(labels: np.ndarray, predictions: np.ndarray):
    """Converts gt and dt boxes to the format used by Prophesee.
    Args:
        gt_boxes: Ground truth boxes.
        dt_boxes: Predicted boxes.
    """

    labels_prophesee = np.zeros((len(labels),), dtype=BBOX_DTYPE)
    predictions_prophesee = np.zeros((len(predictions),), dtype=BBOX_DTYPE)

    for name in BBOX_DTYPE.names:
        if name == "t":
            labels_prophesee[name] = np.asarray(labels["frame"], dtype=BBOX_DTYPE[name])
            predictions_prophesee[name] = np.asarray(predictions["frame"], dtype=BBOX_DTYPE[name])
        elif name == "track_id" and min(labels[name]) == 0:
            labels_prophesee[name] = np.asarray(labels[name] + 1, dtype=BBOX_DTYPE[name])
            predictions_prophesee[name] = np.asarray(predictions[name], dtype=BBOX_DTYPE[name])
        else:
            labels_prophesee[name] = np.asarray(labels[name], dtype=BBOX_DTYPE[name])
            predictions_prophesee[name] = np.asarray(predictions[name], dtype=BBOX_DTYPE[name])

    for i in len(predictions_prophesee):
        padding = calculate_padding(predictions_prophesee[i]["w"], predictions_prophesee[i]["h"])
        predictions_prophesee[i]["x"] -= padding["x"]
        predictions_prophesee[i]["y"] -= padding["y"]
        predictions_prophesee[i]["w"] += padding["w"]
        predictions_prophesee[i]["h"] += padding["h"]

    return labels_prophesee, predictions_prophesee


def main():
    test_label = np.load(
        "/home/efranc/pytracking/pytracking/experiments/rts/rts50/2024_01_10_162004_focus_000/labels.npy"
    )
    test_prediction = np.load(
        "/home/efranc/pytracking/pytracking/experiments/rts/rts50/2024_01_10_162004_focus_000/predictions_rgb.npy"
    )
    labels_p, predictions_p = to_prophesee([test_label], [test_prediction])
    print(labels_p.shape)
    print(predictions_p.shape)


if __name__ == "__main__":
    main()
