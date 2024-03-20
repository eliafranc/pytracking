import os
import json
import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
from PIL import Image
from pathlib import Path

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
        class_name = sequence_name.split("-")[0]
        anno_path = "{}/{}/labels_events_left.npy".format(self.base_path, sequence_name)

        ground_truth = np.load(anno_path)
        starting_frame = ground_truth[0]["frame"] + self.frame_offset
        ending_frame = ground_truth[-1]["frame"]
        ground_truth = ground_truth[ground_truth["frame"] >= starting_frame]
        ground_truth_rect = self._preprocess_annotations(ground_truth, starting_frame, ending_frame)

        frames_path = "{}/{}/{}/frames".format(self.base_path, sequence_name)

        frames_list = [
            "{}/{:06d}.jpg".format(frames_path, frame_number)
            for frame_number in range(1, ground_truth_rect.shape[0] + 1)
        ]

        target_class = class_name

        return Sequence(
            sequence_name, frames_list, "evdrone", ground_truth_rect.reshape(-1, 4), object_class=target_class
        )

    def _preprocess_annotations(self, annotations, start_frame, end_frame):
        annotations = annotations[annotations["frame"] >= start_frame]
        # TODO: Watch out for multi object tracking ...
        for i in range(start_frame, end_frame + 1):
            if i not in annotations["frame"]:
                annotations = np.append(annotations, np.array([(i, 0, 0, 1, 1)], dtype=annotations.dtype))
        return annotations

    @staticmethod
    def _load_mask(path):
        if not path.exists():
            print("Error: Could not read: ", path, flush=True)
            return None
        im = np.array(Image.open(path))
        im = np.atleast_3d(im)[..., 0]
        return im

    def _get_anno_frame_path(self, seq_path, frame_name):
        return os.path.join(seq_path, frame_name)  # frames start from 1

    def __len__(self):
        return len(self.sequence_list)

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
        return sequence_list
