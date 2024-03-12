import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker

"""
Test Files:

'/home/efranc/data/2024_01_10_112814_drone_000/frames_ts.csv',
'/home/efranc/data/2024_01_10_112814_drone_000/frames',
'/home/efranc/data/2024_01_10_112814_drone_000/events_left_final.h5',
'/home/efranc/data/2024_01_10_112814_drone_000/Projection_rgb_to_event_left.npy',
'/home/efranc/data/2024_01_10_112814_drone_000/labels_events_left.npy'

"""

def run_tensor(tracker_name, tracker_param, timings_file, rgb_frame_dir, event_file, homography_file, label_file,  debug=None, save_results=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        ...
    """
    tracker = Tracker(tracker_name, tracker_param)
    tracker.run_on_tensor(timings_file, rgb_frame_dir, event_file, homography_file, label_file , debug=debug)

def main():
    """
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('timings_file', type=str, help='path to a the timings file.')
    parser.add_argument('rgb_frame_dir', type=str, help='path to a the rgb frame dir.')
    parser.add_argument('event_file', type=str, help='path to a the event (hdf5) file.')
    parser.add_argument('homography_file', type=str, help='path to the homography file (npy).')
    parser.add_argument('label_file', type=str, help='path to the label file (npy).')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_tensor(args.tracker_name, args.tracker_param, args.timings_file, args.rgb_frame_dir, args.event_file, args.homography_file, args.label_file, args.debug, args.save_results)
    """
    run_tensor('rts', 'rts50', '/home/efranc/data/2024_01_10_112814_drone_000/frames_ts.csv',
        '/home/efranc/data/2024_01_10_112814_drone_000/frames',
        '/home/efranc/data/2024_01_10_112814_drone_000/events_left_final.h5',
        '/home/efranc/data/2024_01_10_112814_drone_000/Projection_rgb_to_events_left.npy',
        '/home/efranc/data/2024_01_10_112814_drone_000/labels_events_left.npy', None, True)


if __name__ == '__main__':
    main()
