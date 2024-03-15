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


def to_prophesee(labels_list: list, predictions_list: list):
    """Converts gt and dt boxes to the format used by Prophesee.
    Args:
        gt_boxes: Ground truth boxes.
        dt_boxes: Predicted boxes.
    """
    assert len(labels_list) == len(predictions_list)
    labels_list_prophesee = []
    predictions_list_prophesee = []

    # As our labels and predictions do not have a timestamp, rather a frame number, we will use the frame number for t

    for sequence_label, sequence_prediction in zip(labels_list, predictions_list):

        labels_prophesee = np.zeros((len(sequence_label),), dtype=BBOX_DTYPE)
        predictions_prophesee = np.zeros((len(sequence_prediction),), dtype=BBOX_DTYPE)

        #TODO: At some point need to fix the offset for initial bounding box (4 per obj_id) for the labels

        for name in BBOX_DTYPE.names:
            if name == 't':
                labels_prophesee[name] = np.asarray(sequence_label['frame'], dtype=BBOX_DTYPE[name])
                predictions_prophesee[name] = np.asarray(sequence_prediction['frame'], dtype=BBOX_DTYPE[name])
            elif name == 'track_id' and min(sequence_label[name]) == 0:
                labels_prophesee[name] = np.asarray(sequence_label[name] + 1, dtype=BBOX_DTYPE[name])
                predictions_prophesee[name] = np.asarray(sequence_prediction[name], dtype=BBOX_DTYPE[name])
            else:
                labels_prophesee[name] = np.asarray(sequence_label[name], dtype=BBOX_DTYPE[name])
                predictions_prophesee[name] = np.asarray(sequence_prediction[name], dtype=BBOX_DTYPE[name])

        labels_list_prophesee.append(labels_prophesee)
        predictions_list_prophesee.append(predictions_prophesee)

    return labels_prophesee, predictions_prophesee

def main():
    test_label = np.load('/home/efranc/data/2024_01_10_162004_focus_000/labels_events_left.npy')
    test_prediction = np.load('/home/efranc/pytracking/pytracking/experiments/rts/rts50/2024_01_10_162004_focus_000/predictions.npy')
    labels_p, predictions_p = to_prophesee([test_label], [test_prediction])
    print(labels_p.shape)
    print(predictions_p.shape)

if __name__ == "__main__":
    main()