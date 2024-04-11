import importlib
import os
import time
from collections import OrderedDict
from pathlib import Path

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from evutils.io.reader import EventReader_HDF5
from ltr.data.bounding_box_utils import masks_to_bboxes
from pytracking.evaluation.environment import env_settings
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
from pytracking.utils.convert_vot_anno_to_rect import convert_vot_anno_to_rect
from pytracking.utils.plotting import draw_figure, overlay_mask
from pytracking.utils.visdom import Visdom

_tracker_disp_colors = {
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 0, 0),
    4: (255, 255, 255),
    5: (0, 0, 0),
    6: (0, 255, 128),
    7: (123, 123, 123),
    8: (255, 128, 0),
    9: (128, 0, 255),
}


def trackerlist(name: str, parameter_name: str, run_ids=None, display_name: str = None):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, run_id, display_name) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, run_id: int = None, display_name: str = None):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = "{}/{}/{}".format(env.results_path, self.name, self.parameter_name)
            self.segmentation_dir = "{}/{}/{}".format(env.segmentation_path, self.name, self.parameter_name)
        else:
            self.results_dir = "{}/{}/{}_{:03d}".format(env.results_path, self.name, self.parameter_name, self.run_id)
            self.segmentation_dir = "{}/{}/{}_{:03d}".format(
                env.segmentation_path, self.name, self.parameter_name, self.run_id
            )

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tracker", self.name))
        if os.path.isdir(tracker_module_abspath):
            tracker_module = importlib.import_module("pytracking.tracker.{}".format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.visdom = None

    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        if debug > 0 and visdom_info.get("use_visdom", True):
            try:
                self.visdom = Visdom(
                    debug, {"handler": self._visdom_ui_handler, "win_id": "Tracking"}, visdom_info=visdom_info
                )

                # Show help
                help_text = (
                    "You can pause/unpause the tracker by pressing "
                    "space"
                    " with the "
                    "Tracking"
                    " window "
                    "selected. During paused mode, you can track for one frame by pressing the right arrow key."
                    "To enable/disable plotting of a data block, tick/untick the corresponding entry in "
                    "block list."
                )
                self.visdom.register(help_text, "text", 1, "Help")
            except:
                time.sleep(0.5)
                print(
                    "!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n"
                    "!!! Start Visdom in a separate terminal window by typing 'visdom' !!!"
                )

    def _visdom_ui_handler(self, data):
        if data["event_type"] == "KeyPress":
            if data["key"] == " ":
                self.pause_mode = not self.pause_mode

            elif data["key"] == "ArrowRight" and self.pause_mode:
                self.step = True

    def create_tracker(self, params):
        tracker = self.tracker_class(params)
        tracker.visdom = self.visdom
        return tracker

    def run_sequence(self, seq, visualization=None, debug=None, visdom_info=None, multiobj_mode=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            visdom_info: Visdom info.
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()
        visualization_ = visualization

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, "debug", 0)
        if visualization is None:
            if debug is None:
                visualization_ = getattr(params, "visualization", False)
            else:
                visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)
        if visualization_ and self.visdom is None:
            self.init_visualization()

        # Get init information
        init_info = seq.init_info()
        is_single_object = not seq.multiobj_mode

        if multiobj_mode is None:
            multiobj_mode = getattr(params, "multiobj_mode", getattr(self.tracker_class, "multiobj_mode", "default"))

        if multiobj_mode == "default" or is_single_object:
            tracker = self.create_tracker(params)
        elif multiobj_mode == "parallel":
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom)
        else:
            raise ValueError("Unknown multi object mode {}".format(multiobj_mode))

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def run_tensor_sequence(
        self, seq, delta_t=10, rgb_only=False, visualization=None, debug=None, visdom_info=None, multiobj_mode=None
    ):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            visdom_info: Visdom info.
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()
        visualization_ = visualization

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, "debug", 0)
        if visualization is None:
            if debug is None:
                visualization_ = getattr(params, "visualization", False)
            else:
                visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)
        if visualization_ and self.visdom is None:
            self.init_visualization()

        # Get init information
        is_single_object = False

        if multiobj_mode is None:
            multiobj_mode = getattr(params, "multiobj_mode", getattr(self.tracker_class, "multiobj_mode", "default"))

        if multiobj_mode == "default" or is_single_object:
            tracker = self.create_tracker(params)
        elif multiobj_mode == "parallel":
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom)
        else:
            raise ValueError("Unknown multi object mode {}".format(multiobj_mode))

        output = self._track_tensor_sequence(tracker, seq, delta_t, rgb_only)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i
        # segmentation[i] is the segmentation mask for frame i (numpy array)

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i
        # segmentation[i] is the multi-label segmentation mask for frame i (numpy array)

        output = {"target_bbox": [], "time": [], "segmentation": [], "object_presence_score": []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        if tracker.params.visualization and self.visdom is None:
            self.visualize(image, init_info.get("init_bbox"))

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)

        init_default = {
            "target_bbox": init_info.get("init_bbox"),
            "clf_target_bbox": init_info.get("init_bbox"),
            "time": time.time() - start_time,
            "segmentation": init_info.get("init_mask"),
            "object_presence_score": 1.0,
        }

        _store_outputs(out, init_default)

        segmentation = out["segmentation"] if "segmentation" in out else None
        bboxes = [init_default["target_bbox"]]
        if "clf_target_bbox" in out:
            bboxes.append(out["clf_target_bbox"])
        if "clf_search_area" in out:
            bboxes.append(out["clf_search_area"])
        if "segm_search_area" in out:
            bboxes.append(out["segm_search_area"])

        if self.visdom is not None:
            tracker.visdom_draw_tracking(image, bboxes, segmentation)
        elif tracker.params.visualization:
            self.visualize(image, bboxes, segmentation)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            while True:
                if not self.pause_mode:
                    break
                elif self.step:
                    self.step = False
                    break
                else:
                    time.sleep(0.1)

            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info["previous_output"] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {"time": time.time() - start_time})

            segmentation = out["segmentation"] if "segmentation" in out else None

            bboxes = [out["target_bbox"]]
            if "clf_target_bbox" in out:
                bboxes.append(out["clf_target_bbox"])
            if "clf_search_area" in out:
                bboxes.append(out["clf_search_area"])
            if "segm_search_area" in out:
                bboxes.append(out["segm_search_area"])

            if self.visdom is not None:
                tracker.visdom_draw_tracking(image, bboxes, segmentation)
            elif tracker.params.visualization:
                self.visualize(image, bboxes, segmentation)

        for key in ["target_bbox", "segmentation"]:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        # next two lines are needed for oxuva output format.
        output["image_shape"] = image.shape[:2]
        output["object_presence_score_threshold"] = tracker.params.get("object_presence_score_threshold", 0.55)

        return output

    def _track_tensor_sequence(self, tracker, seq, delta_t, rgb_only):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i
        # segmentation[i] is the segmentation mask for frame i (numpy array)

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i
        # segmentation[i] is the multi-label segmentation mask for frame i (numpy array)

        output = {"target_bbox": [], "time": [], "segmentation": [], "object_presence_score": []}
        sequence_name = seq["sequence_name"]
        timings_file = seq["timings_file"]
        rgb_frame_dir = seq["rgb_frame_dir"]
        event_file = seq["event_file"]
        homography_file = seq["homography_file"]
        label_file = seq["label_file"]

        def _create_tensor(
            frame_number,
            rgb_frame_dir,
            rgb_frames,
            event_reader,
            homography,
            timings,
            gray_channel=0,
            dt_ms=10,
            rgb_only=False,
        ):
            rgb_frame = cv.imread(os.path.join(rgb_frame_dir, rgb_frames[frame_number]))
            warped_rgb_image = cv.warpPerspective(rgb_frame, homography, (1280, 720))
            if rgb_only:
                return warped_rgb_image
            gray_warped_rgb_image = cv.cvtColor(warped_rgb_image, cv.COLOR_BGR2GRAY)
            start_ts = int(np.floor(timings["t"][frame_number] / 1000))
            events = event_reader.read(start_ts, start_ts + dt_ms)
            # 10 Me/s (10^7 e/s) cutoff threshold
            ev_threshhold = 10000000 * (dt_ms / 1000)
            if events.shape[0] >= ev_threshhold or events.size == 0:
                return cv.merge([gray_warped_rgb_image, gray_warped_rgb_image, gray_warped_rgb_image])

            on_events = events[events["p"] == 1]
            off_events = events[events["p"] == 0]
            on_frame = np.zeros((720, 1280), dtype=np.uint8)
            on_frame[on_events["y"], on_events["x"]] = 255
            off_frame = np.zeros((720, 1280), dtype=np.uint8)
            off_frame[off_events["y"], off_events["x"]] = 255
            if gray_channel == 0:
                final_image = cv.merge([gray_warped_rgb_image, on_frame, off_frame])
            elif gray_channel == 1:
                final_image = cv.merge([on_frame, gray_warped_rgb_image, off_frame])
            elif gray_channel == 2:
                final_image = cv.merge([on_frame, off_frame, gray_warped_rgb_image])
            else:
                merged_image = cv.merge([gray_warped_rgb_image, gray_warped_rgb_image, gray_warped_rgb_image])
                merged_image[on_frame > 0] = [255, 0, 0]
                merged_image[off_frame > 0] = [0, 0, 255]
                final_image = merged_image

            return final_image

        def _init_bbox_from_labels(labels, init_frames_for_track_id, obj_id):
            obj_specific_labels = labels[labels["track_id"] == obj_id - 1]
            initial_frame = obj_specific_labels[obj_specific_labels["frame"] == init_frames_for_track_id[obj_id]]
            if initial_frame.size == 0:
                return None
            x = int(np.clip(initial_frame["x"], 0, 1280))
            y = int(np.clip(initial_frame["y"], 0, 720))
            w = int(np.clip(initial_frame["w"], 0, 1280))
            h = int(np.clip(initial_frame["h"], 0, 720))
            return [x, y, w, h]

        def _get_tracker_init_dictionaries(init_frames_for_track_id, labels):
            init_bbox = OrderedDict()
            init_object_ids = []
            sequence_object_ids = []
            lowest_frame_number = np.min(list(init_frames_for_track_id.values()))
            for obj_id, value in init_frames_for_track_id.items():
                # Only initialize objects that appear in the first frame
                if value == lowest_frame_number:
                    init_object_ids.append(obj_id)
                    sequence_object_ids.append(obj_id)
                    init_bbox[obj_id] = _init_bbox_from_labels(labels, init_frames_for_track_id, obj_id)

            return init_bbox, init_object_ids, init_object_ids, sequence_object_ids

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Load necessary files
        timings = np.genfromtxt(timings_file, delimiter=",", names=True)
        rgb_frames = sorted(os.listdir(rgb_frame_dir))
        event_reader = EventReader_HDF5(event_file)
        homography = np.load(homography_file)
        labels = np.load(label_file)
        labels["track_id"] = labels["track_id"] - min(labels["track_id"])

        # Get initial frame numbers and indices for labels for each object
        unique_track_ids = np.unique([label["track_id"] for label in labels])
        init_frames_for_track_id = {}
        inital_label_offset = 4
        for track_id in unique_track_ids:
            # Set key for dictionary to start with 1 instead of 0 as it is necessary for the tracker to work
            init_frames_for_track_id[track_id + 1] = int(
                labels[np.where(labels["track_id"] == track_id)]["frame"][0] + inital_label_offset
            )

        # Get the initial tensor for the first frame
        initial_tensor = _create_tensor(
            init_frames_for_track_id[1],
            rgb_frame_dir,
            rgb_frames,
            event_reader,
            homography,
            timings,
            3,
            delta_t,
            rgb_only,
        )
        init_bb, init_obj_ids, obj_ids, sequence_obj_ids = _get_tracker_init_dictionaries(
            init_frames_for_track_id, labels
        )

        # Create init_info dictionary
        init_info = {
            "init_bbox": init_bb,
            "init_object_ids": init_obj_ids,
            "object_ids": obj_ids,
            "sequence_object_ids": sequence_obj_ids,
        }

        if tracker.params.visualization and self.visdom is None:
            self.visualize(initial_tensor, init_info.get("init_bbox"))

        start_time = time.time()
        out = tracker.initialize(initial_tensor, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)

        init_default = {
            "target_bbox": init_info.get("init_bbox"),
            "clf_target_bbox": init_info.get("init_bbox"),
            "time": time.time() - start_time,
            "segmentation": init_info.get("init_mask"),
            "object_presence_score": 1.0,
        }

        _store_outputs(out, init_default)

        segmentation = out["segmentation"] if "segmentation" in out else None
        bboxes = [init_default["target_bbox"]]
        if "clf_target_bbox" in out:
            bboxes.append(out["clf_target_bbox"])
        if "clf_search_area" in out:
            bboxes.append(out["clf_search_area"])
        if "segm_search_area" in out:
            bboxes.append(out["segm_search_area"])

        if self.visdom is not None:
            tracker.visdom_draw_tracking(initial_tensor, bboxes, segmentation)
        elif tracker.params.visualization:
            self.visualize(initial_tensor, bboxes, segmentation)

        start_frame = int(init_frames_for_track_id[1] + 1)

        for frame_num, rgb_frame in enumerate(rgb_frames[start_frame:], start=start_frame):
            while True:
                if not self.pause_mode:
                    break
                elif self.step:
                    self.step = False
                    break
                else:
                    time.sleep(0.1)

            tensor = _create_tensor(
                frame_num, rgb_frame_dir, rgb_frames, event_reader, homography, timings, 3, delta_t, rgb_only
            )

            info = OrderedDict()
            info["previous_output"] = prev_output
            if len(unique_track_ids) > len(sequence_obj_ids):
                new_init_obj_ids = []
                new_init_bboxes = OrderedDict()
                for not_yet_init_ids in np.setdiff1d(unique_track_ids + 1, sequence_obj_ids):
                    if frame_num == init_frames_for_track_id[not_yet_init_ids]:
                        # init_frames_for_track_id[not_yet_init_ids] = frame_num
                        bbox = _init_bbox_from_labels(labels, init_frames_for_track_id, not_yet_init_ids)
                        if bbox is not None:
                            new_init_obj_ids.append(not_yet_init_ids)
                            new_init_bboxes[not_yet_init_ids] = bbox
                        else:
                            init_frames_for_track_id[not_yet_init_ids] = frame_num - 1

                # If any new objects are to be tracked, initialize the tracker with the new objects
                if len(new_init_obj_ids) > 0:
                    info["init_object_ids"] = new_init_obj_ids
                    info["init_bbox"] = new_init_bboxes
                    sequence_obj_ids.extend(new_init_obj_ids)

            info["sequence_object_ids"] = sequence_obj_ids
            start_time = time.time()
            out = tracker.track(tensor, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {"time": time.time() - start_time})

            segmentation = out["segmentation"] if "segmentation" in out else None

            bboxes = [out["target_bbox"]]
            if "clf_target_bbox" in out:
                bboxes.append(out["clf_target_bbox"])
            if "clf_search_area" in out:
                bboxes.append(out["clf_search_area"])
            if "segm_search_area" in out:
                bboxes.append(out["segm_search_area"])

            if self.visdom is not None:
                tracker.visdom_draw_tracking(tensor, bboxes, segmentation)
            elif tracker.params.visualization:
                self.visualize(tensor, bboxes, segmentation)

        for key in ["target_bbox", "segmentation"]:
            if key in output and len(output[key]) <= 1:
                print(key)
                output.pop(key)

        # next two lines are needed for oxuva output format.
        output["image_shape"] = tensor.shape[:2]
        output["object_presence_score_threshold"] = tracker.params.get("object_presence_score_threshold", 0.55)

        return output

    def run_video_generic(
        self, debug=None, visdom_info=None, videofilepath=None, optional_box=None, save_results=False
    ):
        """Run the tracker with the webcam or a provided video file.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, "debug", 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, "multiobj_mode", getattr(self.tracker_class, "multiobj_mode", "default"))

        if multiobj_mode == "default":
            tracker = self.create_tracker(params)
            if hasattr(tracker, "initialize_features"):
                tracker.initialize_features()
        elif multiobj_mode == "parallel":
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=False)
        else:
            raise ValueError("Unknown multi object mode {}".format(multiobj_mode))

        class UIControl:
            def __init__(self):
                self.mode = "init"  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.new_init = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == "init":
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = "select"
                elif event == cv.EVENT_MOUSEMOVE and self.mode == "select":
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == "select":
                    self.target_br = (x, y)
                    self.mode = "init"
                    self.new_init = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
                return bb

        ui_control = UIControl()

        display_name = "Display: " + self.name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        frame_number = 0

        if videofilepath is not None:
            assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
            ", videofilepath must be a valid videofile"
            cap = cv.VideoCapture(videofilepath)
            ret, frame = cap.read()
            frame_number += 1
            cv.imshow(display_name, frame)
        else:
            cap = cv.VideoCapture(0)

        next_object_id = 1
        sequence_object_ids = []
        prev_output = OrderedDict()
        output_boxes = OrderedDict()

        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's format is [x,y,w,h]"

            out = tracker.initialize(
                frame,
                {
                    "init_bbox": OrderedDict({next_object_id: optional_box}),
                    "init_object_ids": [
                        next_object_id,
                    ],
                    "object_ids": [
                        next_object_id,
                    ],
                    "sequence_object_ids": [
                        next_object_id,
                    ],
                },
            )

            prev_output = OrderedDict(out)

            output_boxes[next_object_id] = [
                optional_box,
            ]
            sequence_object_ids.append(next_object_id)
            next_object_id += 1

        # Wait for initial bounding box if video!
        paused = videofilepath is not None

        while True:

            if not paused:
                # Capture frame-by-frame
                ret, frame = cap.read()
                frame_number += 1
                if frame is None:
                    break

            frame_disp = frame.copy()

            info = OrderedDict()
            info["previous_output"] = prev_output

            if ui_control.new_init:
                ui_control.new_init = False
                init_state = ui_control.get_bb()

                info["init_object_ids"] = [
                    next_object_id,
                ]
                info["init_bbox"] = OrderedDict({next_object_id: init_state})
                sequence_object_ids.append(next_object_id)
                if save_results:
                    output_boxes[next_object_id] = [
                        init_state,
                    ]
                next_object_id += 1

            # Draw box
            if ui_control.mode == "select":
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)

            if len(sequence_object_ids) > 0:
                info["sequence_object_ids"] = sequence_object_ids
                out = tracker.track(frame, info)

                prev_output = OrderedDict(out)

                if "segmentation" in out:
                    frame_disp = overlay_mask(frame_disp, out["segmentation"])
                    mask_image = np.zeros(frame_disp.shape, dtype=frame_disp.dtype)

                    if save_results:
                        mask_image = overlay_mask(mask_image, out["segmentation"])
                        if not os.path.exists(self.results_dir):
                            os.makedirs(self.results_dir)
                        cv.imwrite(self.results_dir + f"seg_{frame_number}.jpg", mask_image)

                if "target_bbox" in out:
                    for obj_id, state in out["target_bbox"].items():
                        state = [int(s) for s in state]
                        cv.rectangle(
                            frame_disp,
                            (state[0], state[1]),
                            (state[2] + state[0], state[3] + state[1]),
                            _tracker_disp_colors[obj_id],
                            5,
                        )
                        if save_results:
                            output_boxes[obj_id].append(state)

            # Put text
            font_color = (255, 255, 255)
            msg = "Select target(s). Press 'r' to reset or 'q' to quit."
            cv.rectangle(frame_disp, (5, 5), (630, 40), (50, 50, 50), -1)
            cv.putText(frame_disp, msg, (10, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 2)

            if videofilepath is not None:
                msg = "Press SPACE to pause/resume the video."
                cv.rectangle(frame_disp, (5, 50), (530, 90), (50, 50, 50), -1)
                cv.putText(frame_disp, msg, (10, 75), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 2)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("r"):
                next_object_id = 1
                sequence_object_ids = []
                prev_output = OrderedDict()

                info = OrderedDict()

                info["object_ids"] = []
                info["init_object_ids"] = []
                info["init_bbox"] = OrderedDict()
                tracker.initialize(frame, info)
                ui_control.mode = "init"
            # 'Space' to pause video
            elif key == 32 and videofilepath is not None:
                paused = not paused

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = "webcam" if videofilepath is None else Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, "video_{}".format(video_name))
            print(f"Save results to: {base_results_path}")
            for obj_id, bbox in output_boxes.items():
                tracked_bb = np.array(bbox).astype(int)
                bbox_file = "{}_{}.txt".format(base_results_path, obj_id)
                np.savetxt(bbox_file, tracked_bb, delimiter="\t", fmt="%d")

    def run_video_noninteractive(
        self, debug=None, visdom_info=None, videofilepath=None, optional_box=None, save_results=True
    ):
        """Run the tracker with a provided video file. Output the bounding
        boxes in form of an ordered dictionary that contains a list of
        bounding boxes for each object id.

        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, "debug", 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, "multiobj_mode", getattr(self.tracker_class, "multiobj_mode", "default"))

        if multiobj_mode == "default":
            tracker = self.create_tracker(params)
            if hasattr(tracker, "initialize_features"):
                tracker.initialize_features()
        elif multiobj_mode == "parallel":
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=False)
        else:
            raise ValueError("Unknown multi object mode {}".format(multiobj_mode))

        frame_number = 0

        if videofilepath is not None:
            assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
            ", videofilepath must be a valid videofile"
            cap = cv.VideoCapture(videofilepath)
            _, frame = cap.read()
            frame_number += 1
        else:
            cap = cv.VideoCapture(0)

        next_object_id = 1
        sequence_object_ids = []
        prev_output = OrderedDict()
        output_boxes = OrderedDict()
        output_masks = OrderedDict()

        assert optional_box is not None, "No initial bounding box provided."
        assert isinstance(optional_box, (list, tuple))
        assert len(optional_box) == 4, "valid box's format is [x,y,w,h]"

        out = tracker.initialize(
            frame,
            {
                "init_bbox": OrderedDict({next_object_id: optional_box}),
                "init_object_ids": [
                    next_object_id,
                ],
                "object_ids": [
                    next_object_id,
                ],
                "sequence_object_ids": [
                    next_object_id,
                ],
            },
        )

        prev_output = OrderedDict(out)

        output_boxes[next_object_id] = [
            optional_box,
        ]
        output_masks[next_object_id] = [
            None,
        ]
        sequence_object_ids.append(next_object_id)
        next_object_id += 1

        # Wait for initial bounding box if video!
        paused = videofilepath is not None

        while True:

            if not paused:
                # Capture frame-by-frame
                _, frame = cap.read()
                frame_number += 1
                if frame is None:
                    break

            info = OrderedDict()
            info["previous_output"] = prev_output

            if len(sequence_object_ids) > 0:
                info["sequence_object_ids"] = sequence_object_ids
                out = tracker.track(frame, info)

                prev_output = OrderedDict(out)

                if "target_bbox" in out:
                    for obj_id, state in out["target_bbox"].items():
                        state = [int(s) for s in state]
                        output_boxes[obj_id].append(state)

                if "segmentation" in out:
                    output_masks[obj_id].append(out["segmentation"])

            # After first frame, let the tracker run on the rest of the video.
            paused = False
            if frame_number == 10:
                break

        # When everything done, release the capture
        cap.release()

        return output_boxes

    def run_on_tensor(
        self,
        sequence_name,
        timings_file,
        rgb_frame_dir,
        event_file,
        homography_file,
        label_file,
        delta_t=10,
        debug=None,
        vis=False,
        rgb_only=False,
        save_results=False,
    ):
        """
        Run the tracker on a sequence (rgb frames, events for each frame). Output the bounding
        boxes in form of an ordered dictionary that contains a list of
        bounding boxes for each object id.

        args:
            debug: Debug level.
        """

        _rgb_ev_tracker_disp_colors = {0: (255, 162, 0), 1: (101, 208, 119)}

        def _create_tensor(
            frame_number,
            rgb_frame_dir,
            rgb_frames,
            event_reader,
            homography,
            timings,
            gray_channel=0,
            dt_ms=10,
            rgb_only=False,
        ):
            rgb_frame = cv.imread(os.path.join(rgb_frame_dir, rgb_frames[frame_number]))
            warped_rgb_image = cv.warpPerspective(rgb_frame, homography, (1280, 720))
            if rgb_only:
                return warped_rgb_image
            gray_warped_rgb_image = cv.cvtColor(warped_rgb_image, cv.COLOR_BGR2GRAY)
            start_ts = int(np.floor(timings["t"][frame_number] / 1000))
            events = event_reader.read(start_ts, start_ts + dt_ms)
            if events.shape[0] >= 50000:
                return cv.merge([gray_warped_rgb_image, gray_warped_rgb_image, gray_warped_rgb_image])

            on_events = events[events["p"] == 1]
            off_events = events[events["p"] == 0]
            on_frame = np.zeros((720, 1280), dtype=np.uint8)
            on_frame[on_events["y"], on_events["x"]] = 255
            off_frame = np.zeros((720, 1280), dtype=np.uint8)
            off_frame[off_events["y"], off_events["x"]] = 255
            if gray_channel == 0:
                final_image = cv.merge([gray_warped_rgb_image, on_frame, off_frame])
            elif gray_channel == 1:
                final_image = cv.merge([on_frame, gray_warped_rgb_image, off_frame])
            elif gray_channel == 2:
                final_image = cv.merge([on_frame, off_frame, gray_warped_rgb_image])
            else:
                merged_image = cv.merge([gray_warped_rgb_image, gray_warped_rgb_image, gray_warped_rgb_image])
                merged_image[on_frame > 0] = [255, 0, 0]
                merged_image[off_frame > 0] = [0, 0, 255]
                final_image = merged_image

            return final_image

        def _init_bbox_from_labels(labels, init_frames_for_track_id, obj_id):
            obj_specific_labels = labels[labels["track_id"] == obj_id - 1]
            initial_frame = obj_specific_labels[obj_specific_labels["frame"] == init_frames_for_track_id[obj_id]]
            x = int(np.clip(initial_frame["x"], 0, 1280))
            y = int(np.clip(initial_frame["y"], 0, 720))
            w = int(np.clip(initial_frame["w"], 0, 1280))
            h = int(np.clip(initial_frame["h"], 0, 720))
            return [x, y, w, h]

        def _get_tracker_init_dictionaries(init_frames_for_track_id, labels):
            init_bbox = OrderedDict()
            init_object_ids = []
            sequence_object_ids = []
            lowest_frame_number = np.min(list(init_frames_for_track_id.values()))
            for obj_id, value in init_frames_for_track_id.items():
                # Only initialize objects that appear in the first frame
                if value == lowest_frame_number:
                    init_object_ids.append(obj_id)
                    sequence_object_ids.append(obj_id)
                    init_bbox[obj_id] = _init_bbox_from_labels(labels, init_frames_for_track_id, obj_id)

            return init_bbox, init_object_ids, init_object_ids, sequence_object_ids

        def _save_results_to_npy(output_boxes, output_path, rgb_only, delta_t):
            if rgb_only:
                file_path = os.path.join(output_path, "predictions_rgb.npy")
            else:
                file_path = os.path.join(output_path, f"predictions_{delta_t}ms.npy")
            output_array = np.zeros(
                sum(len(v) for v in output_boxes.values()),
                dtype=np.dtype(
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
                ),
            )
            index = 0
            for frame, bbox_dict in output_boxes.items():
                for track_id, (bbox, score) in bbox_dict.items():
                    output_array[index] = (frame, track_id, bbox[0], bbox[1], bbox[2], bbox[3], score, 1, 1)
                    index += 1
            np.save(file_path, output_array)

        # Tracker parameter setup
        params = self.get_parameters()
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, "debug", 0)
        params.debug = debug_
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        self._init_visdom({"use_visdom": True}, debug_)
        multiobj_mode = getattr(params, "multiobj_mode", getattr(self.tracker_class, "multiobj_mode", "default"))

        if multiobj_mode == "default":
            tracker = self.create_tracker(params)
            if hasattr(tracker, "initialize_features"):
                tracker.initialize_features()
        elif multiobj_mode == "parallel":
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=False)
        else:
            raise ValueError("Unknown multi object mode {}".format(multiobj_mode))

        # Setup output directories
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        sequence_output_dir = os.path.join(self.results_dir, sequence_name)
        if save_results or vis:
            if not os.path.exists(sequence_output_dir):
                os.makedirs(sequence_output_dir)
            if vis:
                if rgb_only:
                    vis_output_dir = os.path.join(sequence_output_dir, "frames_rgb")
                else:
                    vis_output_dir = os.path.join(sequence_output_dir, f"frames_{delta_t}ms")
                if not os.path.exists(vis_output_dir):
                    os.makedirs(vis_output_dir)

        # Load necessary files
        timings = np.genfromtxt(timings_file, delimiter=",", names=True)
        rgb_frames = sorted(os.listdir(rgb_frame_dir))
        event_reader = EventReader_HDF5(event_file)
        homography = np.load(homography_file)
        labels = np.load(label_file)

        # Get initial frame numbers and indices for labels for each object
        unique_track_ids = np.unique([label["track_id"] for label in labels])
        init_frames_for_track_id = {}
        inital_label_offset = 4
        for track_id in unique_track_ids:
            # Set key for dictionary to start with 1 instead of 0 as it is necessary for the tracker to work
            init_frames_for_track_id[track_id + 1] = int(
                labels[np.where(labels["track_id"] == track_id)]["frame"][0] + inital_label_offset
            )

        # Get the initial tensor for the first frame
        initial_tensor = _create_tensor(
            init_frames_for_track_id[min(init_frames_for_track_id.keys())],
            rgb_frame_dir,
            rgb_frames,
            event_reader,
            homography,
            timings,
            3,
            delta_t,
            rgb_only,
        )

        # Set up dictionaries and lists for tracking
        output_boxes = OrderedDict()
        output_masks = OrderedDict()
        init_bb, init_obj_ids, obj_ids, sequence_obj_ids = _get_tracker_init_dictionaries(
            init_frames_for_track_id, labels
        )

        # Initialize tracker for objects appearing first
        out = tracker.initialize(
            initial_tensor,
            {
                "init_bbox": init_bb,
                "init_object_ids": init_obj_ids,
                "object_ids": obj_ids,
                "sequence_object_ids": sequence_obj_ids,
            },
        )

        # Fill in the initial output dictionaries and visualize if set
        prev_output = OrderedDict(out)
        output_boxes[init_frames_for_track_id[min(init_frames_for_track_id.keys())]] = OrderedDict()
        output_masks[init_frames_for_track_id[min(init_frames_for_track_id.keys())]] = OrderedDict()
        for obj_id, bbox in init_bb.items():
            output_boxes[init_frames_for_track_id[min(init_frames_for_track_id.keys())]][obj_id] = (bbox, 1)
            output_masks[init_frames_for_track_id[min(init_frames_for_track_id.keys())]][obj_id] = None
            if vis:
                # initial_tensor = cv.rectangle(
                #     initial_tensor,
                #     (bbox[0], bbox[1]),
                #     (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                #     _rgb_ev_tracker_disp_colors[int(rgb_only)],
                #     2,
                # )
                cv.imwrite(
                    f"{vis_output_dir}/{init_frames_for_track_id[min(init_frames_for_track_id.keys())]:06d}.jpg",
                    initial_tensor,
                )

        # Set up variable regarding frame numbers
        last_frame = len(rgb_frames) - 1
        current_frame = int(init_frames_for_track_id[min(init_frames_for_track_id.keys())] + 1)

        while True:
            print(current_frame)
            tensor = _create_tensor(
                current_frame, rgb_frame_dir, rgb_frames, event_reader, homography, timings, 3, delta_t, rgb_only
            )
            if tensor is None:
                break

            info = OrderedDict()
            info["previous_output"] = prev_output
            output_boxes[current_frame] = OrderedDict()
            output_masks[current_frame] = OrderedDict()
            for obj_id in sequence_obj_ids:
                output_boxes[current_frame][obj_id] = None
                output_masks[current_frame][obj_id] = None

            # Check if there are any new objects to track
            if len(unique_track_ids) > len(sequence_obj_ids):
                for not_yet_init_ids in np.setdiff1d(unique_track_ids + 1, sequence_obj_ids):
                    new_init_obj_ids = []
                    new_init_bboxes = OrderedDict()
                    if current_frame == init_frames_for_track_id.get(not_yet_init_ids):
                        bbox = _init_bbox_from_labels(labels, init_frames_for_track_id, not_yet_init_ids)
                        new_init_obj_ids.append(not_yet_init_ids)
                        new_init_bboxes[not_yet_init_ids] = bbox
                        sequence_obj_ids.append(not_yet_init_ids)
                        output_boxes[current_frame][not_yet_init_ids] = (bbox, 1)
                        output_masks[current_frame][not_yet_init_ids] = None

                # If any new objects are to be tracked, initialize the tracker with the new objects
                if len(new_init_obj_ids) > 0:
                    info["init_object_ids"] = new_init_obj_ids
                    info["init_bbox"] = new_init_bboxes

            if len(sequence_obj_ids) > 0:
                info["sequence_object_ids"] = sequence_obj_ids

                # Track objects that are present in the current frame
                out = tracker.track(tensor, info)
                prev_output = OrderedDict(out)
                bboxes_for_vis = []

                if "target_bbox" in out:
                    for obj_id, state in sorted(out["target_bbox"].items()):
                        state = [int(s) for s in state]

                        # If bounding box infeasible remove respective dict entry
                        if not np.all(np.asarray(state) == np.asarray([0, 0, 1, 1])):
                            bboxes_for_vis.append((obj_id, state, out["score"][obj_id]))
                            output_boxes[current_frame][obj_id] = (state, out["score"][obj_id])

                    for obj_id in sequence_obj_ids:
                        if output_boxes[current_frame][obj_id] == None:
                            del output_boxes[current_frame][obj_id]

                    if "segmentation" in out:
                        output_masks[current_frame][obj_id] = out["segmentation"]

                    if vis:
                        # for obj_id, bbox, score in bboxes_for_vis:
                        #     tensor = cv.rectangle(
                        #         tensor,
                        #         (bbox[0], bbox[1]),
                        #         (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                        #         _rgb_ev_tracker_disp_colors[int(rgb_only)],
                        #         2,
                        #     )
                        # cv.putText(
                        #     tensor,
                        #     str(round(score, 3)),
                        #     (bbox[0], bbox[1] - 7),
                        #     cv.FONT_HERSHEY_SIMPLEX,
                        #     0.4,
                        #     _tracker_disp_colors[obj_id],
                        #     1,
                        # )

                        cv.imwrite(f"{vis_output_dir}/{current_frame:06d}.jpg", tensor)

            # Break tracking loop if all objects have ended
            if current_frame == last_frame:
                break

            current_frame += 1

        # Save results if set
        if save_results:
            _save_results_to_npy(output_boxes, sequence_output_dir, rgb_only, delta_t)

        return output_boxes

    def run_vot2020(self, debug=None, visdom_info=None):
        params = self.get_parameters()
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.run_id = self.run_id

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, "debug", 0)

        if debug is None:
            visualization_ = getattr(params, "visualization", False)
        else:
            visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)

        tracker = self.create_tracker(params)
        tracker.initialize_features()

        output_segmentation = tracker.predicts_segmentation_mask()

        import pytracking.evaluation.vot2020 as vot

        def _convert_anno_to_list(vot_anno):
            vot_anno = [vot_anno[0], vot_anno[1], vot_anno[2], vot_anno[3]]
            return vot_anno

        def _convert_image_path(image_path):
            return image_path

        """Run tracker on VOT."""

        if output_segmentation:
            handle = vot.VOT("mask")
        else:
            handle = vot.VOT("rectangle")

        vot_anno = handle.region()

        image_path = handle.frame()
        if not image_path:
            return
        image_path = _convert_image_path(image_path)

        image = self._read_image(image_path)

        if output_segmentation:
            vot_anno_mask = vot.make_full_size(vot_anno, (image.shape[1], image.shape[0]))
            bbox = masks_to_bboxes(torch.from_numpy(vot_anno_mask), fmt="t").squeeze().tolist()
        else:
            bbox = _convert_anno_to_list(vot_anno)
            vot_anno_mask = None

        out = tracker.initialize(image, {"init_mask": vot_anno_mask, "init_bbox": bbox})

        if out is None:
            out = {}
        prev_output = OrderedDict(out)

        # Track
        while True:
            image_path = handle.frame()
            if not image_path:
                break
            image_path = _convert_image_path(image_path)

            image = self._read_image(image_path)

            info = OrderedDict()
            info["previous_output"] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)

            if output_segmentation:
                pred = out["segmentation"].astype(np.uint8)
            else:
                state = out["target_bbox"]
                pred = vot.Rectangle(*state)
            handle.report(pred, 1.0)

            segmentation = out["segmentation"] if "segmentation" in out else None
            if self.visdom is not None:
                tracker.visdom_draw_tracking(image, out["target_bbox"], segmentation)
            elif tracker.params.visualization:
                self.visualize(image, out["target_bbox"], segmentation)

    def run_vot(self, debug=None, visdom_info=None):
        params = self.get_parameters()
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        params.run_id = self.run_id

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, "debug", 0)

        if debug is None:
            visualization_ = getattr(params, "visualization", False)
        else:
            visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)

        tracker = self.create_tracker(params)
        tracker.initialize_features()

        import pytracking.evaluation.vot as vot

        def _convert_anno_to_list(vot_anno):
            vot_anno = [
                vot_anno[0][0][0],
                vot_anno[0][0][1],
                vot_anno[0][1][0],
                vot_anno[0][1][1],
                vot_anno[0][2][0],
                vot_anno[0][2][1],
                vot_anno[0][3][0],
                vot_anno[0][3][1],
            ]
            return vot_anno

        def _convert_image_path(image_path):
            image_path_new = image_path[20:-2]
            return "".join(image_path_new)

        """Run tracker on VOT."""

        handle = vot.VOT("polygon")

        vot_anno_polygon = handle.region()
        vot_anno_polygon = _convert_anno_to_list(vot_anno_polygon)

        init_state = convert_vot_anno_to_rect(vot_anno_polygon, tracker.params.vot_anno_conversion_type)

        image_path = handle.frame()
        if not image_path:
            return
        image_path = _convert_image_path(image_path)

        image = self._read_image(image_path)
        tracker.initialize(image, {"init_bbox": init_state})

        # Track
        while True:
            image_path = handle.frame()
            if not image_path:
                break
            image_path = _convert_image_path(image_path)

            image = self._read_image(image_path)
            out = tracker.track(image)
            state = out["target_bbox"]

            handle.report(vot.Rectangle(state[0], state[1], state[2], state[3]))

            segmentation = out["segmentation"] if "segmentation" in out else None
            if self.visdom is not None:
                tracker.visdom_draw_tracking(image, out["target_bbox"], segmentation)
            elif tracker.params.visualization:
                self.visualize(image, out["target_bbox"], segmentation)

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module("pytracking.parameter.{}.{}".format(self.name, self.parameter_name))
        params = param_module.parameters()
        return params

    def init_visualization(self):
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect("key_press_event", self.press)
        plt.tight_layout()

    def visualize(self, image, state, segmentation=None):
        self.ax.cla()
        self.ax.imshow(image)
        if segmentation is not None:
            self.ax.imshow(segmentation, alpha=0.5)

        if isinstance(state, (OrderedDict, dict)):
            boxes = [v for k, v in state.items()]
        elif isinstance(state, list):
            boxes = state
        else:
            boxes = (state,)

        for i, box in enumerate(boxes, start=1):
            col = _tracker_disp_colors[i]
            col = [float(c) / 255.0 for c in col]
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor=col, facecolor="none")
            self.ax.add_patch(rect)

        if getattr(self, "gt_state", None) is not None:
            gt_state = self.gt_state
            rect = patches.Rectangle(
                (gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor="g", facecolor="none"
            )
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis("equal")
        draw_figure(self.fig)

        if self.pause_mode:
            keypress = False
            while not keypress:
                keypress = plt.waitforbuttonpress()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == "p":
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == "r":
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def _read_image(self, image_file: str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
