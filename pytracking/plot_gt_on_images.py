import os
import sys
import argparse
import numpy as np
import cv2
from glob import glob

env_path = os.path.join(os.path.dirname(__file__), "..")
if env_path not in sys.path:
    sys.path.append(env_path)


def plot_labels_on_single_image(input_image: np.ndarray, labels: np.ndarray, frame_number) -> np.ndarray:
    """Plot the labels on a single image.
    args:
        input_image: Input image.
        labels: Labels to plot.
    returns:
        output_image: Image with labels plotted.
    """
    output_image = input_image.copy()
    for label in labels[labels["frame"] == frame_number]:
        x, y, w, h = label["x"], label["y"], label["w"], label["h"]
        cv2.rectangle(output_image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 2)
    return output_image


def plot_labels_for_directory(directory, labels):
    """Plot the labels on the images.
    args:
        directory: Directory with the images.
        labels: Labels to plot.
    """
    for image in sorted(os.listdir(directory)):
        if not image.endswith(".jpg"):
            continue
        frame_number = int(str.lstrip(image).split(".")[0])
        input_image = cv2.imread(os.path.join(directory, image))
        output_image = plot_labels_on_single_image(input_image, labels, frame_number)
        image_name = image.split(".")[0] + "_labeled.jpg"
        cv2.imwrite(os.path.join(directory, image_name), output_image)


def plot_labels(results_dir: str):
    """Plot the labels on the images.
    args:
        results_dir: Directory with the results.
    """
    for sequence in os.listdir(results_dir):
        if not os.path.isdir(os.path.join(results_dir, sequence)):
            continue
        labels = np.load(os.path.join(results_dir, sequence, "labels.npy"))
        for subsequence in glob(os.path.join(results_dir, sequence, "frames_*")):
            if not os.path.isdir(subsequence):
                continue
            plot_labels_for_directory(subsequence, labels)


def main():
    parser = argparse.ArgumentParser(description="Plot labels on images.")
    parser.add_argument("results_dir", type=str, help="Directory with the results.")
    args = parser.parse_args()
    plot_labels(args.results_dir)


if __name__ == "__main__":
    main()
