import argparse
import os
import sys
from glob import glob

import cv2
import numpy as np

env_path = os.path.join(os.path.dirname(__file__), "..")
if env_path not in sys.path:
    sys.path.append(env_path)

result_dirs = ["results_7dot5meps", "results_10meps", "results_12dot5meps"]
input_configs = ["tomp50_001", "tomp50_002", "tomp50_003", "tomp50_004", "tomp50_005", "tomp50_006", "tomp50_007"]
timings = {"results_7dot5meps": [], "results_10meps": [], "results_12dot5meps": []}


def compute_mean_inference_time_for_sequence(timing_file: str):
    timings = np.loadtxt(timing_file)
    return np.mean(timings)


def compute_mean_inference_time_for_config(config_dir: str):
    timings = []
    timing_files = glob(os.path.join(config_dir, "2024*_time.txt"))
    for timing_file in timing_files:
        sequence_timing_mean = compute_mean_inference_time_for_sequence(timing_file)
        timings.append(sequence_timing_mean)
    print("Mean inference time for config:", config_dir, np.mean(timings))
    return np.mean(timings)


def main():
    root = "/home/efranc/pytracking/pytracking"
    for result_dir in result_dirs:
        for input_config in input_configs:
            config_dir = os.path.join(root, result_dir, "tracking_results", "tomp", input_config)
            mean_inference_time = compute_mean_inference_time_for_config(config_dir)
            timings[result_dir].append(mean_inference_time)

    for result_dir in result_dirs:
        print("Evaluation for", result_dir)
        print(f"Mean inference time for {result_dir}: {np.mean(timings[result_dir])}")

    print("Overall mean inference time:", np.mean([np.mean(timings[result_dir]) for result_dir in result_dirs]))


if __name__ == "__main__":
    main()
