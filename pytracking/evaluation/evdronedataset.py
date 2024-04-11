import numpy as np

from pytracking.evaluation.data import BaseDataset, Sequence, SequenceList

DTYPE = np.dtype(
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


class EvDroneDataset(BaseDataset):
    """
    Event-based drone dataset.

    """

    def __init__(self, frame_offset=4):
        super().__init__()
        self.base_path = self.env_settings.evdrone_path
        self.sequence_list = self._get_sequence_list()
        self.frame_offset = frame_offset

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        base_sequence = sequence_name[:-2]
        object_id = int(sequence_name.split("_")[-1])

        anno_path = "{}/{}/labels_events_left_test.npy".format(self.base_path, base_sequence)
        gt = np.load(anno_path)
        gt["track_id"] = gt["track_id"] - min(gt["track_id"])
        object_specific_gt = gt[gt["track_id"] == object_id - 1]
        starting_frame = int(object_specific_gt[0]["frame"] + self.frame_offset)
        ending_frame = int(object_specific_gt[-1]["frame"])
        object_specific_gt = object_specific_gt[object_specific_gt["frame"] >= starting_frame]
        processed_gt = self._preprocess_annotations(object_specific_gt, starting_frame, ending_frame, base_sequence)

        frames_path = "{}/{}/frames".format(self.base_path, sequence_name[:-2])

        frames_list = [
            "{}/{:06d}.jpg".format(frames_path, frame_number)
            for frame_number in range(starting_frame, ending_frame + 1)
        ]

        return Sequence(sequence_name, frames_list, "evdrone", processed_gt, object_class="drone")

    def _preprocess_annotations(self, annotations, start_frame, end_frame, seq):
        annotations = annotations[annotations["frame"] >= start_frame]
        # Make sure that the object_ids start from 1
        annotations["track_id"] += 1

        new_annotations = []
        for frame in range(start_frame, end_frame + 1):
            if frame not in annotations["frame"]:
                new_annotations.append([0, 0, 0, 0])
                continue
            else:
                annotation = annotations[annotations["frame"] == frame]
                assert len(annotation) == 1
                new_annotations.append(
                    [float(annotation["x"]), float(annotation["y"]), float(annotation["w"]), float(annotation["h"])]
                )
        new_annotations = np.array(new_annotations)

        return new_annotations

    def _inverse_pad_annotation(self, annotation):
        padded_x, padded_y, padded_w, padded_h = annotation
        unpad = self._calculate_inverse_padding(padded_w, padded_h)
        x = padded_x + unpad["x"]
        y = padded_y + unpad["y"]
        w = padded_w - unpad["w"]
        h = padded_h - unpad["h"]
        return np.asarray([x, y, w, h])

    def _calculate_inverse_padding(self, padded_w, padded_h):
        w = int(padded_w / (1 + 2 * 0.08))
        h = int(padded_h / (1 + 2 * 0.1))
        unpad_w = padded_w - w
        unpad_h = padded_h - h
        unpad_x = int(unpad_w / 2)
        unpad_y = int(unpad_h * 0.8)
        return {"x": unpad_x, "y": unpad_y, "w": unpad_w, "h": unpad_h}

    def __len__(self):
        return len(self.sequence_list)

    def _get_invalid_sequences(self):
        invalid_sequences = ["2024_01_10_163531_recording_016_3"]
        invalid_sequences += ["2024_01_10_115957_mavic_003_1", "2024_01_10_115957_mavic_003_2"]
        return invalid_sequences

    def _get_sequence_list(self):
        sequence_list = [
            "2024_01_10_112814_drone_000",
            "2024_01_10_112814_drone_001",
            "2024_01_10_112911_drone_1_30_000",
            "2024_01_10_112957_drone_1_repeat_000",
            "2024_01_10_112957_drone_1_repeat_001",
            "2024_01_10_113124_drone_2_actual_000",
            "2024_01_10_113124_drone_2_actual_001",
            "2024_01_10_113124_drone_2_actual_002",
            "2024_01_10_113124_drone_2_actual_003",
            "2024_01_10_112814_drone_002",
            "2024_01_10_112911_drone_1_30_001",
            "2024_01_10_113124_drone_2_actual_004",
            "2024_01_10_113242_recording_000",
            "2024_01_10_113242_recording_001",
            "2024_01_10_113242_recording_002",
            "2024_01_10_113242_recording_003",
            "2024_01_10_113329_max_000",
            "2024_01_10_113513_drone_25_000",
            "2024_01_10_113513_drone_25_001",
            "2024_01_10_113545_40_kmh_drone_000",
            "2024_01_10_115151_test_big_drone_000",
            "2024_01_10_115151_test_big_drone_001",
            "2024_01_10_115151_test_big_drone_002",
            "2024_01_10_115151_test_big_drone_003",
            "2024_01_10_115435_recording_000",
            "2024_01_10_115608_20km_000",
            "2024_01_10_115639_return_000",
            "2024_01_10_115714_40km_000",
            "2024_01_10_115957_mavic_000",
            "2024_01_10_115957_mavic_001",
            "2024_01_10_115957_mavic_002",
            "2024_01_10_115957_mavic_003",
            "2024_01_10_120253_mavic_000",
            "2024_01_10_120253_mavic_001",
            "2024_01_10_120253_mavic_002",
            "2024_01_10_120253_mavic_003",
            "2024_01_10_120347_10km_000",
            "2024_01_10_120347_10km_001",
            "2024_01_10_120429_20_000",
            "2024_01_10_120442_recording_000",
            "2024_01_10_120509_40_000",
            "2024_01_10_120509_40_001",
            "2024_01_10_120554_recording_000",
            "2024_01_10_153109_dji_000",
            "2024_01_10_153109_dji_001",
            "2024_01_10_153109_dji_002",
            "2024_01_10_153109_dji_003",
            "2024_01_10_153109_dji_004",
            "2024_01_10_153109_dji_005",
            "2024_01_10_153109_dji_006",
            "2024_01_10_153109_dji_007",
            "2024_01_10_153501_dji_slankg_000",
            "2024_01_10_153501_dji_slankg_001",
            "2024_01_10_153501_dji_slankg_002",
            "2024_01_10_153501_dji_slankg_003",
            "2024_01_10_153501_dji_slankg_004",
            "2024_01_10_153740_dark_background_dji_000",
            "2024_01_10_153740_dark_background_dji_001",
            "2024_01_10_153740_dark_background_dji_002",
            "2024_01_10_153740_dark_background_dji_003",
            "2024_01_10_153740_dark_background_dji_004",
            "2024_01_10_153740_dark_background_dji_005",
            "2024_01_10_153740_dark_background_dji_006",
            "2024_01_10_153740_dark_background_dji_007",
            "2024_01_10_153740_dark_background_dji_008",
            "2024_01_10_153740_dark_background_dji_009",
            "2024_01_10_155201_moving_car__003",
            "2024_01_10_155504_try_two_000",
            "2024_01_10_155757_try_3_001",
            "2024_01_10_160626_back_drive_000",
            "2024_01_10_160626_back_drive_001",
            "2024_01_10_160626_back_drive_002",
            "2024_01_10_160626_back_drive_003",
            "2024_01_10_160626_back_drive_004",
            "2024_01_10_160626_back_drive_005",
            "2024_01_10_160626_back_drive_006",
            "2024_01_10_160626_back_drive_007",
            "2024_01_10_160626_back_drive_008",
            "2024_01_10_160626_back_drive_009",
            "2024_01_10_160626_back_drive_010",
            "2024_01_10_160626_back_drive_011",
            "2024_01_10_160626_back_drive_012",
            "2024_01_10_160626_back_drive_013",
            "2024_01_10_160626_back_drive_014",
            "2024_01_10_160626_back_drive_015",
            "2024_01_10_160626_back_drive_016",
            "2024_01_10_160626_back_drive_017",
            "2024_01_10_160626_back_drive_018",
            "2024_01_10_160626_back_drive_019",
            "2024_01_10_160626_back_drive_020",
            "2024_01_10_160626_back_drive_021",
            "2024_01_10_160626_back_drive_022",
            "2024_01_10_162004_focus_000",
            "2024_01_10_162004_focus_001",
            "2024_01_10_162004_focus_002",
            "2024_01_10_162004_focus_003",
            "2024_01_10_162004_focus_004",
            "2024_01_10_162004_focus_005",
            "2024_01_10_162004_focus_006",
            "2024_01_10_162004_focus_007",
            "2024_01_10_162004_focus_008",
            "2024_01_10_162004_focus_009",
            "2024_01_10_162004_focus_010",
            "2024_01_10_162004_focus_011",
            "2024_01_10_162004_focus_012",
            "2024_01_10_162004_focus_013",
            "2024_01_10_162004_focus_014",
            "2024_01_10_162004_focus_015",
            "2024_01_10_162004_focus_016",
            "2024_01_10_162004_focus_017",
            "2024_01_10_162004_focus_018",
            "2024_01_10_162004_focus_019",
            "2024_01_10_162004_focus_020",
            "2024_01_10_162004_focus_021",
            "2024_01_10_162004_focus_022",
            "2024_01_10_162004_focus_023",
            "2024_01_10_162004_focus_024",
            "2024_01_10_162004_focus_025",
            "2024_01_10_162614_recording_000",
            "2024_01_10_162614_recording_001",
            "2024_01_10_162614_recording_002",
            "2024_01_10_162614_recording_003",
            "2024_01_10_162614_recording_004",
            "2024_01_10_162614_recording_005",
            "2024_01_10_162614_recording_006",
            "2024_01_10_162614_recording_007",
            "2024_01_10_162614_recording_008",
            "2024_01_10_162614_recording_009",
            "2024_01_10_162614_recording_010",
            "2024_01_10_162614_recording_011",
            "2024_01_10_162614_recording_012",
            "2024_01_10_162614_recording_013",
            "2024_01_10_162614_recording_014",
            "2024_01_10_162614_recording_015",
            "2024_01_10_162614_recording_016",
            "2024_01_10_162614_recording_017",
            "2024_01_10_162614_recording_018",
            "2024_01_10_162614_recording_019",
            "2024_01_10_162614_recording_020",
            "2024_01_10_162614_recording_021",
            "2024_01_10_162614_recording_022",
            "2024_01_10_162614_recording_023",
            "2024_01_10_162614_recording_024",
            "2024_01_10_162614_recording_025",
            "2024_01_10_162614_recording_026",
            "2024_01_10_163234_recording_000",
            "2024_01_10_163234_recording_001",
            "2024_01_10_163234_recording_002",
            "2024_01_10_163234_recording_003",
            "2024_01_10_163234_recording_004",
            "2024_01_10_163234_recording_005",
            "2024_01_10_163234_recording_006",
            "2024_01_10_163234_recording_007",
            "2024_01_10_163234_recording_008",
            "2024_01_10_163234_recording_009",
            "2024_01_10_163234_recording_010",
            "2024_01_10_163234_recording_011",
            "2024_01_10_163234_recording_012",
            "2024_01_10_163234_recording_013",
            "2024_01_10_163531_recording_000",
            "2024_01_10_163531_recording_001",
            "2024_01_10_163531_recording_002",
            "2024_01_10_163531_recording_003",
            "2024_01_10_163531_recording_004",
            "2024_01_10_163531_recording_005",
            "2024_01_10_163531_recording_006",
            "2024_01_10_163531_recording_007",
            "2024_01_10_163531_recording_008",
            "2024_01_10_163531_recording_009",
            "2024_01_10_163531_recording_010",
            "2024_01_10_163531_recording_011",
            "2024_01_10_163531_recording_012",
            "2024_01_10_163531_recording_013",
            "2024_01_10_163531_recording_014",
            "2024_01_10_163531_recording_015",
            "2024_01_10_163531_recording_016",
            "2024_01_10_163531_recording_017",
            "2024_01_10_163531_recording_018",
            "2024_01_10_163531_recording_019",
            "2024_01_10_163531_recording_020",
            "2024_01_10_163531_recording_021",
            "2024_01_10_163531_recording_022",
            "2024_01_10_163531_recording_023",
            "2024_01_10_164255_recording_000",
            "2024_01_10_164255_recording_001",
            "2024_01_10_164255_recording_002",
            "2024_01_10_164255_recording_003",
            "2024_01_10_164255_recording_004",
            "2024_01_10_164255_recording_005",
            "2024_01_10_164255_recording_006",
            "2024_01_10_164624_recording_000",
            "2024_01_10_164624_recording_001",
            "2024_01_10_164624_recording_002",
            "2024_01_10_164624_recording_003",
            "2024_01_10_164624_recording_004",
            "2024_01_10_164624_recording_005",
            "2024_01_10_164624_recording_006",
            "2024_01_10_165031_recording_000",
            "2024_01_10_165031_recording_001",
            "2024_01_10_165031_recording_002",
            "2024_01_10_165031_recording_003",
            "2024_01_10_165031_recording_004",
            "2024_01_10_165031_recording_005",
            "2024_01_10_165031_recording_006",
            "2024_01_10_165031_recording_007",
            "2024_01_10_165031_recording_008",
            "2024_01_10_165031_recording_009",
            "2024_01_10_165031_recording_010",
            "2024_01_10_165031_recording_011",
            "2024_01_10_165031_recording_012",
            "2024_01_10_165031_recording_013",
            "2024_01_10_165031_recording_014",
            "2024_01_10_165826_recording_000",
            "2024_01_10_165826_recording_001",
            "2024_01_10_165826_recording_002",
            "2024_01_10_170509_himo_000",
            "2024_01_10_170509_himo_001",
            "2024_01_10_170509_himo_002",
            "2024_01_10_170509_himo_003",
            "2024_01_10_170509_himo_004",
            "2024_01_10_170509_himo_005",
            "2024_01_10_170509_himo_006",
            "2024_01_10_170509_himo_007",
            "2024_01_10_170509_himo_008",
            "2024_01_10_170509_himo_009",
            "2024_01_10_170509_himo_010",
            "2024_01_10_170509_himo_011",
            "2024_01_10_170911_recording_000",
            "2024_01_10_170911_recording_001",
            "2024_01_10_170911_recording_002",
            "2024_01_10_170911_recording_003",
            "2024_01_10_170911_recording_004",
            "2024_01_10_170911_recording_005",
            "2024_01_10_170911_recording_006",
            "2024_01_10_170911_recording_007",
            "2024_01_10_170911_recording_008",
            "2024_01_10_170911_recording_009",
            "2024_01_10_170911_recording_010",
            "2024_01_10_170911_recording_011",
            "2024_01_10_170911_recording_012",
            "2024_01_10_170911_recording_013",
            "2024_01_10_170911_recording_014",
            "2024_01_10_170911_recording_015",
            "2024_01_10_170911_recording_016",
            "2024_01_10_170911_recording_017",
            "2024_01_10_170911_recording_018",
            "2024_01_10_170911_recording_019",
            "2024_01_10_170911_recording_020",
            "2024_01_10_170911_recording_021",
            "2024_01_10_170911_recording_022",
            "2024_01_10_170911_recording_023",
            "2024_01_10_170911_recording_024",
            "2024_01_10_170911_recording_025",
            "2024_01_10_170911_recording_026",
            "2024_01_10_170911_recording_027",
            "2024_01_10_170911_recording_028",
            "2024_01_10_170911_recording_029",
        ]

        # Make sure each object in a sequence has a unique gt
        new_sequence_list = []
        for seq in sequence_list:
            anno_path = "{}/{}/labels_events_left.npy".format(self.base_path, seq)
            object_ids = np.unique(np.load(anno_path)["track_id"])
            object_ids = object_ids - min(object_ids)
            for object_id in object_ids + 1:
                if f"{seq}_{object_id}" in self._get_invalid_sequences():
                    continue
                new_sequence_list.append(f"{seq}_{object_id}")

        return new_sequence_list
